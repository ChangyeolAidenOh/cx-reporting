"""NLP Stage 3: Reassign BERTopic outliers (-1) by embedding similarity.

HDBSCAN marks low-density docs as outliers even when they are semantically
close to an existing topic. This stage reassigns an outlier only when its
cosine similarity to the nearest topic centroid exceeds `threshold`;
low-similarity docs (ad spam, off-topic) stay -1.

Reassigned docs that land on the product-VoC parent topics (0, 100) are
routed through the stage-2 models so they receive a product-level sub-topic.

Usage:
    python -m analysis.reduce_outliers
    python -m analysis.reduce_outliers --threshold 0.55
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from config.db import get_conn


def load_models():
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("jhgan/ko-sroberta-multitask")
    stage1 = BERTopic.load("models/bertopic_insurance", embedding_model=emb)
    stage2_0 = BERTopic.load("models/bertopic_stage2_0", embedding_model=emb)
    stage2_100 = BERTopic.load("models/bertopic_stage2_100", embedding_model=emb)
    return emb, stage1, stage2_0, stage2_100


def topic_centroids(model, embeddings_by_topic):
    """Mean embedding per topic from the model's training docs."""
    return {t: v.mean(axis=0) for t, v in embeddings_by_topic.items()}


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def run(threshold=0.55):
    emb, stage1, stage2_0, stage2_100 = load_models()

    # 1. Build stage-1 topic centroids from currently assigned (non-outlier) docs
    with get_conn() as conn:
        assigned = pd.read_sql(
            """
            SELECT id, text_clean, topic_id FROM staging.content_enriched
            WHERE topic_id != -1 ORDER BY id
            """, conn)
        outliers = pd.read_sql(
            "SELECT id, text_clean FROM staging.content_enriched WHERE topic_id = -1 ORDER BY id",
            conn)
    print(f"Assigned docs: {len(assigned)} | Outliers: {len(outliers)}")

    # Map stage-2 ids back to stage-1 parent for centroid building
    def to_stage1(t):
        if 200 <= t < 300 or t == 100:
            return 0
        if 100 <= t < 200:
            return 0
        return t
    assigned["s1"] = assigned["topic_id"].apply(to_stage1)

    print("Embedding assigned docs for centroids...")
    emb_assigned = emb.encode(assigned["text_clean"].tolist(), batch_size=64, show_progress_bar=True)
    by_topic = {}
    for t, vec in zip(assigned["s1"], emb_assigned):
        by_topic.setdefault(t, []).append(vec)
    centroids = {t: np.vstack(v).mean(axis=0) for t, v in by_topic.items()}
    topic_ids = list(centroids.keys())
    cent_mat = np.vstack([centroids[t] for t in topic_ids])
    cent_norm = cent_mat / (np.linalg.norm(cent_mat, axis=1, keepdims=True) + 1e-9)

    # 2. Embed outliers, find nearest centroid
    print("Embedding outliers...")
    emb_out = emb.encode(outliers["text_clean"].tolist(), batch_size=64, show_progress_bar=True)
    out_norm = emb_out / (np.linalg.norm(emb_out, axis=1, keepdims=True) + 1e-9)
    sims = out_norm @ cent_norm.T
    best_idx = sims.argmax(axis=1)
    best_sim = sims.max(axis=1)

    outliers["new_s1"] = [topic_ids[i] for i in best_idx]
    outliers["sim"] = best_sim
    keep = outliers[outliers["sim"] >= threshold].copy()
    print(f"\nReassignable at threshold {threshold}: {len(keep)} / {len(outliers)} "
          f"({len(keep) / len(outliers) * 100:.1f}%)")
    print(f"  similarity quantiles: p25={np.percentile(best_sim, 25):.3f} "
          f"p50={np.percentile(best_sim, 50):.3f} p75={np.percentile(best_sim, 75):.3f}")

    # 3. Route docs landing on parent topic 0 through stage-2 models
    to_0 = keep[keep["new_s1"] == 0]
    final = {}
    if len(to_0):
        docs0 = to_0["text_clean"].tolist()
        t0, _ = stage2_0.transform(docs0)
        for doc_id, t in zip(to_0["id"], t0):
            final[doc_id] = 0 if t == -1 else 100 + int(t)
        # those that landed on 100 -> stage2_100
        ids100 = [d for d, t in final.items() if t == 100]
        if ids100:
            docs100 = to_0[to_0["id"].isin(ids100)]["text_clean"].tolist()
            t100, _ = stage2_100.transform(docs100)
            for doc_id, t in zip(ids100, t100):
                final[doc_id] = 100 if t == -1 else 200 + int(t)
    for _, r in keep[keep["new_s1"] != 0].iterrows():
        final[r["id"]] = int(r["new_s1"])

    # 4. Update staging (topic_label left NULL -> topic_mapping.py fills it)
    with get_conn() as conn:
        with conn.cursor() as cur:
            for doc_id, t in final.items():
                cur.execute(
                    "UPDATE staging.content_enriched SET topic_id = %s, topic_label = NULL WHERE id = %s",
                    (t, int(doc_id)))
    print(f"Updated {len(final)} docs; {len(outliers) - len(final)} remain -1")

    dist = pd.Series(list(final.values())).value_counts().sort_index()
    print("\nReassignment distribution (top 15):")
    for t, c in dist.head(15).items():
        print(f"  topic {t}: {c}")
    print("\nNext: python -m analysis.topic_mapping")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=0.55)
    a = p.parse_args()
    run(threshold=a.threshold)


if __name__ == "__main__":
    main()
