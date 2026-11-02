"""Distill topic-category assignments into a serve-time classifier (v2).

Design:
  - Train on non-outlier docs only. Outliers (-1) have no local structure to learn.
  - Target = 7 business categories (what the pipeline consumes), not 85 topic ids.
  - Inference returns category + confidence; conf < threshold -> treated as -1
    (outlier / needs review). We report accuracy@coverage, not just mean accuracy.
  - Separately measure outlier recall: fraction of true -1 docs that fall below
    the threshold (they should).

Usage:
    python scripts/build_serve_classifier.py
"""

import sys
import os
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from config.db import get_conn


EMB_NAME = "jhgan/ko-sroberta-multitask"
OUT_DIR = "models/serve"
EMB_CACHE = f"{OUT_DIR}/train_embeddings.npy"
THRESHOLDS = [0.5, 0.6, 0.7, 0.8]


def load_docs():
    with get_conn() as conn:
        df = pd.read_sql(
            "SELECT id, text_clean, topic_id, topic_label FROM staging.content_enriched "
            "WHERE topic_id IS NOT NULL ORDER BY id", conn)
    df["category"] = df["topic_label"]
    return df


def embed(texts):
    if os.path.exists(EMB_CACHE):
        X = np.load(EMB_CACHE).astype(np.float32)
        if len(X) == len(texts):
            print(f"Loaded cached embeddings {X.shape}")
            return X
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer(EMB_NAME)
    X = emb.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(EMB_CACHE, X.astype(np.float16))
    return X.astype(np.float32)


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


def run():
    df = load_docs()
    X_all = embed(df["text_clean"].tolist())

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
        out_conf = m_full.predict_proba(X_out).max(1)
        acc = report(name, y, pred, conf, out_conf)
        results[name] = dict(acc=acc, pred=pred, conf=conf, out_conf=out_conf)

    best = max(results, key=lambda n: results[n]["acc"])
    print(f"\nBest by overall accuracy: {best} ({results[best]['acc']:.1%})")

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
    joblib.dump(model, f"{OUT_DIR}/classifier.joblib")
    size = os.path.getsize(f"{OUT_DIR}/classifier.joblib") / 1e6
    r = results[best]
    meta = {
        "embedding_model": EMB_NAME, "classifier": best, "target": "category",
        "n_train": int(len(y)), "classes": labels,
        "cv_acc": round(float(r["acc"]), 4),
        "acc_at_coverage": {str(t): {"coverage": round(float((r["conf"] >= t).mean()), 4),
                                     "accuracy": round(float((r["pred"][r["conf"] >= t] == y[r["conf"] >= t]).mean()), 4)}
                            for t in THRESHOLDS},
        "outlier_recall": {str(t): round(float((r["out_conf"] < t).mean()), 4) for t in THRESHOLDS},
        "all_cv_acc": {n: round(float(v["acc"]), 4) for n, v in results.items()},
    }
    with open(f"{OUT_DIR}/classifier_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {OUT_DIR}/classifier.joblib ({size:.1f} MB) + classifier_meta.json")


if __name__ == "__main__":
    run()
