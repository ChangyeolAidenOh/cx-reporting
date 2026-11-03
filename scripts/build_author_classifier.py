"""Distill the Haiku/Sonnet author_type labels into a serve-time classifier (v2).

Why: every weekly batch of blog/cafe docs needs author_type and the LLM pass is
slow (~31 min for 8.7k docs). The labels exist for the full corpus, so a
classifier over ko-sroberta embeddings serves most docs; low-confidence docs go
to Haiku as before.

What matters is not overall accuracy:
  - customer precision: a wrong 'customer' pollutes the customer-only tests
  - unrelated precision: 'unrelated' becomes topic_label='noise' (destructive)
Both are reported per confidence threshold (precision among auto-confirmed
docs), and the model is SELECTED by customer precision at OP_THRESHOLD among
candidates with coverage >= MIN_COVERAGE (ties -> unrelated precision).

v2 changes:
  - media (2.4% of docs, P=31% in v1) is merged into advisor for serving by
    default: downstream, both are supplier-side and both are excluded from the
    customer segment. The DB keeps the original 4-way labels.
  - per-source kNN candidate (blog model + cafe model): the two sources have
    very different author mixes, and v1 errors were concentrated in blog.
  - lexical candidate gets the source flag as a feature.
  - per-source customer precision is printed for the selected model.

Training set: blog + cafearticle docs with author_type_src in (haiku, sonnet).
YouTube comments are customer by rule and never go through this model.

Usage:
    python -m scripts.build_author_classifier
    python -m scripts.build_author_classifier --model knn10_src
    python -m scripts.build_author_classifier --keep-media
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from config.db import get_conn
from analysis.embedding_cache import get_embeddings, EMB_NAME
from analysis.author_features import lexical_features, source_flag, PerSourceKNN, LEX_PATTERNS

OUT_DIR = "models/serve"
THRESHOLDS = [0.5, 0.6, 0.7, 0.8]
ALL_CLASSES = ["customer", "advisor", "media", "unrelated"]
OP_THRESHOLD = 0.7
MIN_COVERAGE = 0.70
PREC_TOLERANCE = 0.005   # customer precision within this of the best -> pick higher coverage
# Serving gates per source x predicted class. blog advisor is raised to 0.85 because
# true customers were being auto-confirmed as advisor there (18 rescued per 118 extra LLM docs).
SOURCE_THRESHOLDS = {
    "blog": {"customer": 0.8, "advisor": 0.85, "unrelated": 0.8},
    "cafearticle": {"customer": 0.7, "advisor": 0.7, "unrelated": 0.7},
}


def gate_for(source, pred_class):
    return SOURCE_THRESHOLDS.get(source, {}).get(pred_class, OP_THRESHOLD)


def load_docs():
    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT id, source, title, text_clean, author_type, author_type_src
            FROM staging.content_enriched
            WHERE source IN ('blog', 'cafearticle')
              AND author_type = ANY(%s)
              AND author_type_src IN ('haiku', 'sonnet')
            ORDER BY id
            """, conn, params=(ALL_CLASSES,))
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


def class_precision(y, pred, mask, cls):
    sel = mask & (pred == cls)
    return float((y[sel] == cls).mean()) if sel.any() else float("nan")


def at_threshold(y, pred, conf, t):
    keep = conf >= t
    acc = float((pred[keep] == y[keep]).mean()) if keep.any() else float("nan")
    return {"coverage": float(keep.mean()), "accuracy": acc,
            "customer_precision": class_precision(y, pred, keep, "customer"),
            "unrelated_precision": class_precision(y, pred, keep, "unrelated"),
            "advisor_precision": class_precision(y, pred, keep, "advisor")}


def report(name, y, pred, conf):
    acc = (pred == y).mean()
    print(f"  {name:<12} acc={acc:.1%}")
    for t in THRESHOLDS:
        r = at_threshold(y, pred, conf, t)
        print(f"      @{t}  cov={r['coverage']:.0%}  acc={r['accuracy']:.1%}  "
              f"customer P={r['customer_precision']:.1%}  unrelated P={r['unrelated_precision']:.1%}  "
              f"advisor P={r['advisor_precision']:.1%}")
    return acc


def run(force_model=None, merge_media=True):
    df = load_docs()
    y = df["author_type"].values.copy()
    if merge_media:
        y = np.where(y == "media", "advisor", y)
    classes = sorted(set(y))
    print(f"Train: {len(y)} blog/cafe docs | classes: {classes}" + ("  (media merged into advisor)" if merge_media else ""))
    print("Class balance:", pd.Series(y).value_counts().to_dict())
    print("By source:", df["source"].value_counts().to_dict(), "| label sources:", df["author_type_src"].value_counts().to_dict())

    X_emb = get_embeddings(df["id"].values, df["text_clean"].tolist())
    X_src = source_flag(df["source"].tolist())
    X_lex = lexical_features(df["title"].tolist(), df["text_clean"].tolist())
    scaler = StandardScaler().fit(X_lex)
    X_emb_src = np.hstack([X_emb, X_src])
    X_emb_lex_src = np.hstack([X_emb, scaler.transform(X_lex), X_src])

    cands = {
        "knn10":          (X_emb,         lambda: KNeighborsClassifier(n_neighbors=10, metric="cosine", weights="distance")),
        "knn10_src":      (X_emb_src,     lambda: PerSourceKNN(n_neighbors=10)),
        "knn20_src":      (X_emb_src,     lambda: PerSourceKNN(n_neighbors=20)),
        "logreg":         (X_emb,         lambda: LogisticRegression(max_iter=3000, C=5.0, n_jobs=-1, class_weight="balanced")),
        "logreg_lex_src": (X_emb_lex_src, lambda: LogisticRegression(max_iter=3000, C=5.0, n_jobs=-1, class_weight="balanced")),
    }

    print("\n5-fold CV:")
    results = {}
    for name, (X, mk) in cands.items():
        pred, conf = cv_predict(X, y, mk)
        acc = report(name, y, pred, conf)
        results[name] = dict(acc=acc, pred=pred, conf=conf)

    if force_model:
        best = force_model
    else:
        scored = {n: at_threshold(y, r["pred"], r["conf"], OP_THRESHOLD) for n, r in results.items()}
        eligible = [n for n in scored if scored[n]["coverage"] >= MIN_COVERAGE] or list(scored)
        top = max(scored[n]["customer_precision"] for n in eligible)
        near = [n for n in eligible if scored[n]["customer_precision"] >= top - PREC_TOLERANCE]
        best = max(near, key=lambda n: (scored[n]["coverage"], scored[n]["unrelated_precision"]))
    r = results[best]
    sel = at_threshold(y, r["pred"], r["conf"], OP_THRESHOLD)
    print(f"\nSelected: {best}, @{OP_THRESHOLD} customer P={sel['customer_precision']:.1%}, "
          f"unrelated P={sel['unrelated_precision']:.1%}, coverage {sel['coverage']:.0%}"
          + ("  [forced]" if force_model else ""))

    pred, conf = r["pred"], r["conf"]
    print("\nPer-source x threshold (selected model):")
    per_source = {}
    for s in ["blog", "cafearticle"]:
        m = (df["source"].values == s)
        per_source[s] = {}
        for t in THRESHOLDS:
            keep = m & (conf >= t)
            pc = class_precision(y, pred, keep, "customer")
            pu = class_precision(y, pred, keep, "unrelated")
            cov = keep.sum() / max(m.sum(), 1)
            per_source[s][str(t)] = {"coverage": round(float(cov), 4),
                                     "customer_precision": round(pc, 4) if pc == pc else None,
                                     "unrelated_precision": round(pu, 4) if pu == pu else None}
            print(f"  {s:<12} @{t}  cov={cov:.0%}  customer P={pc:.1%}  unrelated P={pu:.1%}")
    gate = np.array([conf[i] >= gate_for(src, pred[i]) for i, src in enumerate(df["source"].values)])
    print(f"\nWith per-source/class gates {SOURCE_THRESHOLDS}: cov={gate.mean():.0%}  "
          f"customer P={class_precision(y, pred, gate, 'customer'):.1%}  "
          f"unrelated P={class_precision(y, pred, gate, 'unrelated'):.1%}  "
          f"acc={(pred[gate] == y[gate]).mean():.1%}")
    for s_ in ["blog", "cafearticle"]:
        m = df["source"].values == s_
        g = gate & m
        lost = int(((y == "customer") & g & (pred != "customer")).sum())
        print(f"  {s_:<12} auto {int(g.sum())}/{int(m.sum())} ({g.sum() / max(m.sum(), 1):.0%})  "
              f"customer P={class_precision(y, pred, g, 'customer'):.1%}  "
              f"true customers confirmed as other: {lost} ({lost / max(((y == 'customer') & m).sum(), 1):.1%})")

    p, rc, f, s = precision_recall_fscore_support(y, pred, labels=classes, zero_division=0)
    print("\nPer-class (all docs, no threshold):")
    for c, pi, ri, fi, si in zip(classes, p, rc, f, s):
        print(f"  {c:<10} P={pi:.1%} R={ri:.1%} F1={fi:.1%} n={si}")
    cm = confusion_matrix(y, pred, labels=classes)
    print("\nConfusion (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=classes, columns=classes).to_string())

    import joblib
    X_sel, mk = cands[best]
    model = mk(); model.fit(X_sel, y)
    os.makedirs(OUT_DIR, exist_ok=True)
    uses_lex = "lex" in best
    uses_src = "src" in best
    joblib.dump({"model": model, "scaler": scaler if uses_lex else None,
                 "uses_lex": uses_lex, "uses_src": uses_src, "classes": classes},
                f"{OUT_DIR}/author_classifier.joblib")
    meta = {
        "embedding_model": EMB_NAME, "classifier": best, "target": "author_type",
        "classes": classes, "media_merged_into_advisor": bool(merge_media),
        "n_train": int(len(y)), "uses_lex": uses_lex, "uses_src": uses_src,
        "lex_patterns": LEX_PATTERNS if uses_lex else None,
        "op_threshold": OP_THRESHOLD,
        "source_thresholds": SOURCE_THRESHOLDS,
        "per_source": per_source,
        "selection": {"metric": f"customer_precision@{OP_THRESHOLD}", "min_coverage": MIN_COVERAGE,
                      "precision_tolerance": PREC_TOLERANCE},
        "cv_acc": round(float(r["acc"]), 4),
        "at_threshold": {str(t): {k: (round(v, 4) if v == v else None) for k, v in at_threshold(y, pred, conf, t).items()}
                         for t in THRESHOLDS},
        "all_cv_acc": {n: round(float(v["acc"]), 4) for n, v in results.items()},
    }
    with open(f"{OUT_DIR}/author_classifier_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    size = os.path.getsize(f"{OUT_DIR}/author_classifier.joblib") / 1e6
    print(f"\nSaved {OUT_DIR}/author_classifier.joblib ({size:.1f} MB) + author_classifier_meta.json")
    cv_out = pd.DataFrame({"id": df["id"].values, "source": df["source"].values,
                           "label": y, "label_src": df["author_type_src"].values,
                           "pred": pred, "conf": np.round(conf, 4)})
    cv_out.to_csv(f"{OUT_DIR}/author_cv_predictions.csv", index=False, encoding="utf-8-sig")
    print(f"CV predictions written to {OUT_DIR}/author_cv_predictions.csv ({len(cv_out)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["knn10", "knn10_src", "knn20_src", "logreg", "logreg_lex_src"], default=None)
    ap.add_argument("--keep-media", action="store_true")
    args = ap.parse_args()
    run(force_model=args.model, merge_media=not args.keep_media)
