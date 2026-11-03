"""Distill topic-category assignments into a serve-time classifier (v3).

Design:
  - Train on non-outlier docs only. Outliers (-1) have no local structure to learn.
  - Target = business categories (8 incl. 태아·어린이보험), not topic ids.
  - Training labels come from the topic mapping, the catch-all relabel and the
    태아·어린이보험 rule (label_source in mapping/classifier/rule_fetal).
    Docs set to noise by the author filter are EXCLUDED: that noise means
    "author is unrelated", not "topic is noise", and the author axis is served
    by the author classifier. Review/tie-break docs are excluded as well
    (weaker labels). Incremental docs carry topic_id = -2 (labeled by this
    classifier or by the weekly rules, never by BERTopic) and are excluded so
    the model does not train on its own weekly output.
  - Inference returns category + confidence; conf < threshold -> review.
    We report accuracy@coverage, not just mean accuracy, and outlier recall.
  - Model selection uses the operating metric: accuracy among auto-confirmed
    docs at OP_THRESHOLD, among candidates with coverage >= MIN_COVERAGE
    (ties -> higher coverage). Overall accuracy is reported but does not pick.
  - Embeddings come from the shared id-keyed cache (analysis/embedding_cache.py).

Usage:
    python -m scripts.build_serve_classifier
"""

import sys
import os
import json
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from config.db import get_conn
from analysis.embedding_cache import get_embeddings, EMB_NAME

OUT_DIR = "models/serve"
THRESHOLDS = [0.5, 0.6, 0.7, 0.8]
TRAIN_LABEL_SOURCES = ("mapping", "classifier", "rule_fetal")
OP_THRESHOLD = 0.7      # confidence gate used by orchestration.incremental.classify_new
MIN_COVERAGE = 0.85     # a candidate must auto-confirm at least this share at OP_THRESHOLD
ACC_TOLERANCE = 0.003   # candidates within this of the best acc@OP_THRESHOLD tie -> pick higher coverage


def load_docs():
    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT id, text_clean, topic_id, topic_label, label_source
            FROM staging.content_enriched
            WHERE topic_id IS NOT NULL AND topic_id <> -2
              AND COALESCE(label_source, 'mapping') = ANY(%s)
            ORDER BY id
            """, conn, params=(list(TRAIN_LABEL_SOURCES),))
    df["category"] = df["topic_label"]
    return df


def cv_predict(X, y, make_model, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    pred = np.empty(len(y), dtype=object)
    conf = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        m = make_model()
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])
        pred[te] = m.classes_[p.argmax(1)]
        conf[te] = p.max(1)
    return pred, conf


def report(name, y, pred, conf, out_conf):
    acc = (pred == y).mean()
    line = f"  {name:<8} acc={acc:.1%}"
    for t in THRESHOLDS:
        keep = conf >= t
        a = (pred[keep] == y[keep]).mean() if keep.any() else float("nan")
        line += f" | @{t} cov={keep.mean():.0%} acc={a:.1%}"
    line += " | outlier-recall " + " ".join(f"@{t}:{(out_conf < t).mean():.0%}" for t in THRESHOLDS)
    print(line)
    return acc


def acc_cov(r, y, t):
    keep = r["conf"] >= t
    acc = float((r["pred"][keep] == y[keep]).mean()) if keep.any() else float("nan")
    return acc, float(keep.mean())


def run(force_model=None):
    df = load_docs()
    print("Label sources in training set:", df["label_source"].fillna("mapping").value_counts().to_dict())
    X_all = get_embeddings(df["id"].values, df["text_clean"].tolist())

    is_out = df["topic_id"].values == -1
    X, y = X_all[~is_out], df.loc[~is_out, "category"].values
    X_out = X_all[is_out]
    print(f"Train (non-outlier): {len(y)} docs, {len(set(y))} categories | held-out outliers: {len(X_out)}")
    print("Class balance:", pd.Series(y).value_counts().to_dict())

    cands = {
        "knn5":   lambda: KNeighborsClassifier(n_neighbors=5,  metric="cosine", weights="distance"),
        "knn10":  lambda: KNeighborsClassifier(n_neighbors=10, metric="cosine", weights="distance"),
        "knn20":  lambda: KNeighborsClassifier(n_neighbors=20, metric="cosine", weights="distance"),
        "logreg": lambda: LogisticRegression(max_iter=3000, C=10.0, n_jobs=-1),
    }

    print("\n5-fold CV (category level):")
    results = {}
    for name, mk in cands.items():
        pred, conf = cv_predict(X, y, mk)
        m_full = mk(); m_full.fit(X, y)
        out_conf = m_full.predict_proba(X_out).max(1) if len(X_out) else np.zeros(0)
        acc = report(name, y, pred, conf, out_conf)
        results[name] = dict(acc=acc, pred=pred, conf=conf, out_conf=out_conf)

    if force_model:
        best = force_model
    else:
        eligible = [n for n in results if acc_cov(results[n], y, OP_THRESHOLD)[1] >= MIN_COVERAGE]
        pool = eligible or list(results)
        top_acc = max(acc_cov(results[n], y, OP_THRESHOLD)[0] for n in pool)
        near = [n for n in pool if acc_cov(results[n], y, OP_THRESHOLD)[0] >= top_acc - ACC_TOLERANCE]
        best = max(near, key=lambda n: acc_cov(results[n], y, OP_THRESHOLD)[1])
    a, c = acc_cov(results[best], y, OP_THRESHOLD)
    print(f"\nSelected: {best} — acc@{OP_THRESHOLD}={a:.1%} at coverage {c:.0%} "
          f"(overall {results[best]['acc']:.1%})" + ("  [forced]" if force_model else ""))

    labels = sorted(set(y))
    cm = confusion_matrix(y, results[best]["pred"], labels=labels)
    print("\nConfusion (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=labels, columns=labels).to_string())
    off = sorted(((cm[i, j], a, b) for i, a in enumerate(labels) for j, b in enumerate(labels)
                  if i != j and cm[i, j] > 0), reverse=True)
    print("\nTop confusions:")
    for c, a, b in off[:8]:
        print(f"  {a} -> {b}: {c}")

    import joblib
    model = cands[best](); model.fit(X, y)
    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump(model, f"{OUT_DIR}/classifier.joblib")
    size = os.path.getsize(f"{OUT_DIR}/classifier.joblib") / 1e6
    r = results[best]
    meta = {
        "embedding_model": EMB_NAME, "classifier": best, "target": "category",
        "n_train": int(len(y)), "classes": labels,
        "train_label_sources": list(TRAIN_LABEL_SOURCES),
        "cv_acc": round(float(r["acc"]), 4),
        "selection": {"metric": f"acc@{OP_THRESHOLD}", "min_coverage": MIN_COVERAGE, "acc_tolerance": ACC_TOLERANCE},
        "acc_at_coverage": {str(t): {"coverage": round(float((r["conf"] >= t).mean()), 4),
                                     "accuracy": round(float((r["pred"][r["conf"] >= t] == y[r["conf"] >= t]).mean()), 4)}
                            for t in THRESHOLDS},
        "outlier_recall": {str(t): round(float((r["out_conf"] < t).mean()), 4) if len(r["out_conf"]) else None
                           for t in THRESHOLDS},
        "all_cv_acc": {n: round(float(v["acc"]), 4) for n, v in results.items()},
    }
    with open(f"{OUT_DIR}/classifier_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {OUT_DIR}/classifier.joblib ({size:.1f} MB) + classifier_meta.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["knn5", "knn10", "knn20", "logreg"], default=None)
    args = ap.parse_args()
    run(force_model=args.model)
