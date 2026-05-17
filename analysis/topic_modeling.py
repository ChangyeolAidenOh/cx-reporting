"""NLP: BERTopic x LDA ensemble topic classification.

Assigns topic_id and topic_label to staging.content_enriched documents.
Uses BERTopic (ko-sroberta) as primary, LDA (gensim) for cross-validation.

Usage:
    python -m analysis.topic_modeling
    python -m analysis.topic_modeling --min-topic-size 20 --nr-topics 15
"""

import sys
import os
import re
import json
import pickle
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from config.db import get_conn


# ============================================================
# Korean tokenizer
# ============================================================

def get_tokenizer():
    """Initialize kiwipiepy tokenizer."""
    from kiwipiepy import Kiwi
    kiwi = Kiwi()
    return kiwi


def tokenize_korean(text, kiwi, min_len=2):
    """Tokenize Korean text, keep nouns/verbs/adjectives."""
    if not text or not isinstance(text, str):
        return []

    # POS tags to keep: NNG(일반명사), NNP(고유명사), VV(동사), VA(형용사)
    keep_tags = {"NNG", "NNP", "VV", "VA"}

    tokens = []
    for token in kiwi.tokenize(text):
        if token.tag in keep_tags and len(token.form) >= min_len:
            tokens.append(token.form)

    return tokens


# ============================================================
# Stopwords
# ============================================================

INSURANCE_STOPWORDS = [
    "보험", "가입", "상품", "상담", "설계",  # too generic
    "정보", "확인", "내용", "관련", "경우",
    "이용", "안내", "사항", "문의", "제공",
    "부분", "정도", "생각", "사람", "하나",
    "그것", "자체", "무료", "추천", "비교",
    "블로그", "카페", "네이버", "링크", "클릭",
    "광고", "협찬", "포스팅", "작성",
]


# ============================================================
# BERTopic
# ============================================================

def run_bertopic(docs, min_topic_size=15, nr_topics="auto"):
    """Run BERTopic with ko-sroberta embedding."""
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer

    print("Loading ko-sroberta-multitask...")
    embedding_model = SentenceTransformer("jhgan/ko-sroberta-multitask")

    # Custom vectorizer for Korean
    vectorizer = CountVectorizer(
        stop_words=INSURANCE_STOPWORDS,
        min_df=5,
        max_df=0.85,
        ngram_range=(1, 2),
    )

    print("Fitting BERTopic...")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer,
        min_topic_size=min_topic_size,
        nr_topics=nr_topics if nr_topics != "auto" else "auto",
        language="multilingual",
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(docs)

    # Topic info
    topic_info = topic_model.get_topic_info()
    print(f"\nBERTopic results:")
    print(f"  Topics found: {len(topic_info) - 1} (excluding outlier topic -1)")
    print(f"  Outlier docs: {sum(1 for t in topics if t == -1)} "
          f"({sum(1 for t in topics if t == -1) / len(topics) * 100:.1f}%)")

    return topic_model, topics, probs


def extract_bertopic_labels(topic_model):
    """Extract human-readable topic labels from BERTopic."""
    labels = {}
    for topic_id in topic_model.get_topics():
        if topic_id == -1:
            labels[-1] = "outlier"
            continue
        words = topic_model.get_topic(topic_id)
        top_words = [w for w, _ in words[:4]]
        labels[topic_id] = " | ".join(top_words)
    return labels


# ============================================================
# LDA (cross-validation)
# ============================================================

def run_lda(tokenized_docs, num_topics=10):
    """Run LDA with gensim for cross-validation."""
    from gensim import corpora, models

    print(f"\nFitting LDA (num_topics={num_topics})...")

    # Filter empty docs
    valid_docs = [doc for doc in tokenized_docs if len(doc) > 0]

    dictionary = corpora.Dictionary(valid_docs)
    dictionary.filter_extremes(no_below=5, no_above=0.85)
    corpus = [dictionary.doc2bow(doc) for doc in valid_docs]

    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=15,
        random_state=42,
        alpha="auto",
        eta="auto",
    )

    # Coherence
    from gensim.models.coherencemodel import CoherenceModel
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=valid_docs,
        dictionary=dictionary,
        coherence="c_v",
    )
    coherence = coherence_model.get_coherence()
    print(f"  LDA coherence (c_v): {coherence:.4f}")

    # Print topics
    print(f"  LDA topics:")
    for idx, topic in lda_model.print_topics(num_words=5):
        print(f"    Topic {idx}: {topic}")

    return lda_model, dictionary, corpus, coherence


# ============================================================
# Ensemble: cross-validate BERTopic x LDA
# ============================================================

def cross_validate_topics(bertopic_model, lda_model, dictionary, tokenized_docs, bertopic_topics):
    """Compare BERTopic and LDA topic assignments for consistency."""
    # Get LDA topic for each document
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

    agreement_count = 0
    total = 0

    # For each BERTopic topic, find the most similar LDA topic
    bertopic_to_lda = {}

    for bt_topic in set(bertopic_topics):
        if bt_topic == -1:
            continue

        # Get docs assigned to this BERTopic topic
        bt_doc_indices = [i for i, t in enumerate(bertopic_topics) if t == bt_topic]
        if not bt_doc_indices:
            continue

        # Get LDA topic distribution for these docs
        lda_topic_counts = {}
        for idx in bt_doc_indices:
            if idx < len(corpus) and len(corpus[idx]) > 0:
                lda_topics = lda_model.get_document_topics(corpus[idx])
                if lda_topics:
                    top_lda = max(lda_topics, key=lambda x: x[1])[0]
                    lda_topic_counts[top_lda] = lda_topic_counts.get(top_lda, 0) + 1

        if lda_topic_counts:
            most_common_lda = max(lda_topic_counts, key=lda_topic_counts.get)
            match_rate = lda_topic_counts[most_common_lda] / len(bt_doc_indices)
            bertopic_to_lda[bt_topic] = {
                "lda_topic": most_common_lda,
                "match_rate": round(match_rate, 3),
                "sample_size": len(bt_doc_indices),
            }

    print(f"\nBERTopic-LDA cross-validation:")
    for bt, info in sorted(bertopic_to_lda.items()):
        print(f"  BERTopic {bt} -> LDA {info['lda_topic']} "
              f"(match {info['match_rate']:.1%}, n={info['sample_size']})")

    return bertopic_to_lda


# ============================================================
# Save results
# ============================================================

def update_staging(doc_ids, topics, labels):
    """Update staging.content_enriched with topic assignments."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            updated = 0
            for doc_id, topic_id in zip(doc_ids, topics):
                topic_label = labels.get(topic_id, "unknown")
                cur.execute(
                    """
                    UPDATE staging.content_enriched
                    SET topic_id = %s, topic_label = %s
                    WHERE id = %s
                    """,
                    (int(topic_id), topic_label, int(doc_id)),
                )
                updated += 1
    print(f"Updated {updated} rows in staging.content_enriched")


def save_models(bertopic_model, lda_model, dictionary, labels, cross_val):
    """Save models and metadata."""
    os.makedirs("models", exist_ok=True)

    bertopic_model.save("models/bertopic_insurance")
    lda_model.save("models/lda_insurance.model")
    dictionary.save("models/lda_dictionary.dict")

    metadata = {
        "topic_labels": {str(k): v for k, v in labels.items()},
        "cross_validation": {str(k): v for k, v in cross_val.items()},
    }
    with open("models/topic_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Models saved to models/")


# ============================================================
# Main
# ============================================================

def run(min_topic_size=15, nr_topics="auto", lda_num_topics=10):
    # Extract
    with get_conn() as conn:
        df = pd.read_sql(
            "SELECT id, text_clean FROM staging.content_enriched ORDER BY id",
            conn,
        )
    print(f"Loaded {len(df)} documents from staging")

    docs = df["text_clean"].tolist()
    doc_ids = df["id"].tolist()

    # Tokenize for LDA
    print("Tokenizing with kiwipiepy...")
    kiwi = get_tokenizer()
    tokenized = []
    for doc in docs:
        tokens = tokenize_korean(doc, kiwi)
        tokens = [t for t in tokens if t not in INSURANCE_STOPWORDS]
        tokenized.append(tokens)

    # BERTopic
    bertopic_model, topics, probs = run_bertopic(
        docs,
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
    )
    labels = extract_bertopic_labels(bertopic_model)

    # LDA cross-validation
    lda_model, dictionary, corpus, coherence = run_lda(
        tokenized, num_topics=lda_num_topics
    )
    cross_val = cross_validate_topics(
        bertopic_model, lda_model, dictionary, tokenized, topics
    )

    # Update staging
    update_staging(doc_ids, topics, labels)

    # Save models
    save_models(bertopic_model, lda_model, dictionary, labels, cross_val)

    # Summary
    print(f"\nTopic distribution:")
    topic_counts = pd.Series(topics).value_counts().sort_index()
    for topic_id, count in topic_counts.items():
        label = labels.get(topic_id, "?")
        print(f"  Topic {topic_id}: {count} docs -- {label}")

    print(f"\nPhase 3a (topic modeling) complete.")
    print(f"  Next: Phase 3b (sentiment analysis)")


def main():
    parser = argparse.ArgumentParser(description="Topic modeling for insurance VoC")
    parser.add_argument("--min-topic-size", type=int, default=15)
    parser.add_argument("--nr-topics", default="auto")
    parser.add_argument("--lda-num-topics", type=int, default=10)
    args = parser.parse_args()

    nr = int(args.nr_topics) if args.nr_topics != "auto" else "auto"
    run(
        min_topic_size=args.min_topic_size,
        nr_topics=nr,
        lda_num_topics=args.lda_num_topics,
    )


if __name__ == "__main__":
    main()
