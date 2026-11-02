"""Relabel catch-all buckets using a classifier trained on clean topics.

Topics 0 and 100 are stage-2 sub-outliers ("product VoC with no subtopic")
plus docs reassigned from -1. They were mapped to 상품 설계·고객후기 by default,
but 25-40% of them talk about 자동차보험 or 실손·건강보험. A kNN trained on all
OTHER topics decides, per doc, whether it belongs to a specific category with
confidence >= threshold. Confident docs get the new category (topic_id kept,
topic_label changed + relabel flag); others keep the default.

Usage:
    python scripts/relabel_catchall.py --threshold 0.7 --dry-run
    python scripts/relabel_catchall.py --threshold 0.7
"""

import sys
import os
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from config.db import get_conn
from analysis.topic_mapping import TOPIC_CATEGORY

EMB_CACHE = "models/serve/train_embeddings.npy"
CATCHALL = [0, 100]


def run(threshold=0.7, dry_run=False):
    with get_conn() as conn:
        df = pd.read_sql(
            "SELECT id, text_clean, topic_id, topic_label FROM staging.content_enriched "
            "WHERE topic_id IS NOT NULL ORDER BY id", conn)
    df["category"] = df["topic_id"].map(lambda t: TOPIC_CATEGORY.get(int(t), "noise"))

    X = np.load(EMB_CACHE).astype(np.float32)
    assert len(X) == len(df), "embedding cache out of sync with staging — rebuild classifier first"

    is_catch = df["topic_id"].isin(CATCHALL).values
    is_out = (df["topic_id"] == -1).values
    train = ~is_catch & ~is_out
    print(f"Train (clean topics): {train.sum()} | catch-all to relabel: {is_catch.sum()}")

    knn = KNeighborsClassifier(n_neighbors=10, metric="cosine", weights="distance")
    knn.fit(X[train], df.loc[train, "category"].values)

    proba = knn.predict_proba(X[is_catch])
    pred = knn.classes_[proba.argmax(1)]
    conf = proba.max(1)

    catch = df[is_catch].copy()
    catch["pred"] = pred
    catch["conf"] = conf
    catch["relabel"] = (conf >= threshold) & (pred != catch["category"]) & (pred != "noise")

    print(f"\nPredicted category distribution for catch-all docs (conf >= {threshold}):")
    confident = catch[catch["conf"] >= threshold]
    print(confident["pred"].value_counts().to_string())
    print(f"\nBelow threshold (keep default): {(catch['conf'] < threshold).sum()}")
    print(f"Relabel (confident AND different AND not noise): {catch['relabel'].sum()}")

    for cat in ["자동차보험", "실손·건강보험", "보험금 청구·보상"]:
        s = catch[catch["relabel"] & (catch["pred"] == cat)].head(3)
        if len(s):
            print(f"\n  → {cat} samples:")
            for _, r in s.iterrows():
                print(f"    [{r['conf']:.2f}] {r['text_clean'][:90]}")

    if dry_run:
        print("\n[dry-run] no changes written")
        return

    to_update = catch[catch["relabel"]]
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE staging.content_enriched ADD COLUMN IF NOT EXISTS label_source VARCHAR(20)")
            cur.execute("UPDATE staging.content_enriched SET label_source = 'mapping' WHERE label_source IS NULL")
            for _, r in to_update.iterrows():
                cur.execute(
                    "UPDATE staging.content_enriched SET topic_label = %s, label_source = 'classifier' WHERE id = %s",
                    (r["pred"], int(r["id"])))
    print(f"\nUpdated {len(to_update)} rows (label_source='classifier')")
    print("Next: python scripts/build_serve_classifier.py  (retrain on cleaned labels)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    run(threshold=a.threshold, dry_run=a.dry_run)


if __name__ == "__main__":
    main()
