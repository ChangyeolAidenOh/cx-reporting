"""NLP Stage 2: Hierarchical sub-clustering of the dominant product-VoC topic.

Stage 1 BERTopic groups all blog/cafe product posts into one dense cluster
(Topic 0, ~33% of docs) because the genre is homogeneous. This stage re-runs
BERTopic on that cluster only, so product-level subtopics (자동차보험, 실손,
보험금 청구, 대출 ...) become separable.

Sub-topic ids are offset by +100 (100, 101, ...) to avoid colliding with
stage-1 ids. Outliers from stage 2 keep the parent id (0).

Usage:
    python -m analysis.topic_modeling_stage2
    python -m analysis.topic_modeling_stage2 --parent-topic 0 --min-topic-size 40
"""

import sys
import json
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from config.db import get_conn
from analysis.topic_modeling import INSURANCE_STOPWORDS

def run(parent_topic=0, min_topic_size=40, nr_topics="auto", offset=100):
    SUBTOPIC_OFFSET = offset
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer

    with get_conn() as conn:
        df = pd.read_sql(
            "SELECT id, text_clean FROM staging.content_enriched WHERE topic_id = %s ORDER BY id",
            conn, params=(parent_topic,))
    print(f"Stage 2: {len(df)} docs in parent topic {parent_topic}")

    docs = df["text_clean"].tolist()
    doc_ids = df["id"].tolist()

    embedding_model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    vectorizer = CountVectorizer(stop_words=INSURANCE_STOPWORDS, min_df=5,
                                 max_df=0.80, ngram_range=(1, 2))
    model = BERTopic(embedding_model=embedding_model, vectorizer_model=vectorizer,
                     min_topic_size=min_topic_size, nr_topics=nr_topics,
                     language="multilingual", verbose=True)
    topics, _ = model.fit_transform(docs)

    # Build labels; sub-outliers (-1) stay on parent id
    labels = {}
    for t in model.get_topics():
        if t == -1:
            continue
        words = [w for w, _ in model.get_topic(t)[:4]]
        labels[t] = " | ".join(words)

    n_out = sum(1 for t in topics if t == -1)
    print(f"\nStage 2 results: {len(labels)} subtopics, {n_out} sub-outliers "
          f"({n_out / len(topics) * 100:.1f}%) stay on topic {parent_topic}")

    # Update staging
    with get_conn() as conn:
        with conn.cursor() as cur:
            for doc_id, t in zip(doc_ids, topics):
                if t == -1:
                    continue
                cur.execute(
                    "UPDATE staging.content_enriched SET topic_id = %s, topic_label = %s WHERE id = %s",
                    (SUBTOPIC_OFFSET + int(t), labels[t], int(doc_id)))
    print("Updated staging.content_enriched with sub-topic ids")

    # Save
    model.save(f"models/bertopic_stage2_{parent_topic}")
    meta = {"parent_topic": parent_topic, "offset": SUBTOPIC_OFFSET,
            "labels": {str(SUBTOPIC_OFFSET + k): v for k, v in labels.items()}}
    with open(f"models/topic_stage2_{parent_topic}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\nSub-topic distribution:")
    counts = pd.Series(topics).value_counts().sort_index()
    for t, c in counts.items():
        if t == -1:
            print(f"  Topic {parent_topic} (sub-outlier): {c} docs")
        else:
            print(f"  Topic {SUBTOPIC_OFFSET + t}: {c} docs -- {labels[t]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--offset", type=int, default=100)
    p.add_argument("--parent-topic", type=int, default=0)
    p.add_argument("--min-topic-size", type=int, default=40)
    p.add_argument("--nr-topics", default="auto")
    a = p.parse_args()
    nr = int(a.nr_topics) if a.nr_topics != "auto" else "auto"
    run(parent_topic=a.parent_topic, min_topic_size=a.min_topic_size, nr_topics=nr, offset=a.offset)


if __name__ == "__main__":
    main()
