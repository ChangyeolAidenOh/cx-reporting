"""Verify that safetensors serve models reproduce the original topic ids.

Safetensors drops UMAP/HDBSCAN and infers via c-TF-IDF similarity, so
assignments can drift. This samples staging docs with known stage-1 ids
and reports agreement. Target: >= 90% exact match; mismatches should land
on semantically adjacent topics or -1.

Usage:
    python scripts/verify_serve_models.py --n 600
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from config.db import get_conn


def run(n=600, seed=42):
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer

    emb = SentenceTransformer("jhgan/ko-sroberta-multitask")
    serve = BERTopic.load("models/serve/stage1", embedding_model=emb)

    # Stage-1 ids: 0..58 only (exclude stage-2 ids and reassigned outliers whose
    # original stage-1 id we cannot recover exactly)
    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT id, text_clean, topic_id FROM staging.content_enriched
            WHERE topic_id BETWEEN 1 AND 58
            ORDER BY RANDOM() LIMIT %s
            """, conn, params=(n,))
    print(f"Sample: {len(df)} docs with stage-1 ids 1..58")

    pred, _ = serve.transform(df["text_clean"].tolist())
    df["pred"] = pred

    exact = (df["pred"] == df["topic_id"]).mean()
    to_outlier = (df["pred"] == -1).mean()
    print(f"\nExact match     : {exact:.1%}")
    print(f"Predicted -1    : {to_outlier:.1%}")

    mism = df[df["pred"] != df["topic_id"]]
    if not mism.empty:
        print(f"\nTop mismatches (orig -> pred): ")
        pairs = mism.groupby(["topic_id", "pred"]).size().sort_values(ascending=False).head(10)
        for (o, p), c in pairs.items():
            print(f"  {o:>3} -> {p:>3}: {c}")

    verdict = "PASS" if exact >= 0.90 else ("ACCEPTABLE" if exact >= 0.80 else "FAIL")
    print(f"\nVerdict: {verdict} (threshold: >=90% pass, >=80% acceptable)")
    if verdict == "FAIL":
        print("  -> Consider saving with pickle + storing in GitHub Release / HF Hub instead.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=600)
    a = p.parse_args()
    run(n=a.n)


if __name__ == "__main__":
    main()
