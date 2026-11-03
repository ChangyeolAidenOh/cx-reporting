"""Move product-axis docs out of the 상품 설계·고객후기 residual bucket.

Why: after the 태아·어린이보험 split, 상품 설계·고객후기 is the one row the
category classifier gets wrong (recall ~76%): docs there that the model
confidently assigns to 실손·건강보험 / 태아·어린이보험 / 자동차보험 /
보험금 청구·보상 are mostly product-axis docs the manual topic mapping could
not separate. A 0.85 relabel pass was rejected once because 태아보험 posts went
to 자동차보험; with 태아·어린이보험 now a class of its own that failure mode is
gone, so the pass is retried under sample review.

Labels are out-of-fold (5-fold CV), so a doc is never relabeled by a model that
saw it. Relabeled docs get label_source='classifier' and label_conf.

Usage:
    python -m scripts.relabel_product_bucket                 dry-run
    python -m scripts.relabel_product_bucket --conf 0.85     dry-run at another threshold
    python -m scripts.relabel_product_bucket --apply
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from config.db import get_conn
from analysis.embedding_cache import get_embeddings

SOURCE_LABEL = "상품 설계·고객후기"
TARGETS = ["태아·어린이보험", "실손·건강보험", "자동차보험", "보험금 청구·보상"]
TRAIN_LABEL_SOURCES = ("mapping", "classifier", "rule_fetal")
FETAL_HINT = r"태아|어린이|아이|자녀|신생아|임신|출산|산모|굿앤굿|자녀"


def load_train():
    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT id, title, text_clean, topic_id, topic_label, label_source
            FROM staging.content_enriched
            WHERE topic_id IS NOT NULL AND topic_id <> -2 AND topic_id <> -1
              AND COALESCE(label_source, 'mapping') = ANY(%s)
            ORDER BY id
            """, conn, params=(list(TRAIN_LABEL_SOURCES),))
    return df


def oof_predict(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    pred = np.empty(len(y), dtype=object)
    conf = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        m = KNeighborsClassifier(n_neighbors=10, metric="cosine", weights="distance")
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])
        pred[te] = m.classes_[p.argmax(1)]
        conf[te] = p.max(1)
    return pred, conf


def candidates(conf_min):
    df = load_train()
    X = get_embeddings(df["id"].values, df["text_clean"].tolist())
    pred, conf = oof_predict(X, df["topic_label"].values)
    df["pred"], df["conf"] = pred, conf
    c = df[(df["topic_label"] == SOURCE_LABEL) & df["pred"].isin(TARGETS) & (df["conf"] >= conf_min)].copy()
    return df, c


def dry_run(conf_min):
    df, c = candidates(conf_min)
    src = df[df["topic_label"] == SOURCE_LABEL]
    print(f"{SOURCE_LABEL}: {len(src)} docs; out-of-fold disagreement to product categories: "
          f"{int(src['pred'].isin(TARGETS).sum())} total, {len(c)} at conf >= {conf_min}")
    print("\nBy predicted category (conf >= threshold):")
    print(c["pred"].value_counts().to_string())
    print("\nConfidence buckets of all disagreements:")
    d = src[src["pred"].isin(TARGETS)]
    print(pd.cut(d["conf"], [0, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]).value_counts().sort_index().to_string())
    fetal = c[c["pred"] == "태아·어린이보험"]
    if len(fetal):
        hit = fetal["text_clean"].str.contains(FETAL_HINT, regex=True)
        print(f"\n태아·어린이보험 candidates with a fetal/child hint word in text: {int(hit.sum())}/{len(fetal)}")
    print("\nSample (10 per predicted category):")
    for t in TARGETS:
        sub = c[c["pred"] == t]
        if sub.empty:
            continue
        sub = sub.assign(k=sub["id"].astype(str).map(lambda s: __import__("hashlib").md5(s.encode()).hexdigest()))
        print(f"\n--- {SOURCE_LABEL} -> {t} ({len(sub)}) ---")
        for _, r in sub.sort_values("k").head(10).iterrows():
            print(f"[{r['id']}][conf {r['conf']:.2f}] {str(r['title'])[:50]}\n    {r['text_clean'][:150]}\n")


def apply(conf_min):
    _, c = candidates(conf_min)
    with get_conn() as conn:
        with conn.cursor() as cur:
            for _, r in c.iterrows():
                cur.execute(
                    "UPDATE staging.content_enriched SET topic_label=%s, label_source='classifier', "
                    "label_conf=%s, processed_at=NOW() WHERE id=%s",
                    (r["pred"], float(r["conf"]), int(r["id"])))
        conn.commit()
    print(f"Relabeled {len(c)} docs out of {SOURCE_LABEL} (conf >= {conf_min}): "
          + ", ".join(f"{k} {v}" for k, v in c["pred"].value_counts().items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.85)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    if a.apply:
        apply(a.conf)
    else:
        dry_run(a.conf)
