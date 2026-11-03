"""Incremental steps for the weekly pipeline.

Order (see orchestration/weekly_run.py):
  stage_new()            raw rows not yet in staging -> staging (label_source='pending')
  resolve_brands_new()   blog/cafe: title names one other brand -> brand := that brand
  classify_new()         pending docs -> category serve classifier -> category + confidence
                         conf >= 0.7 -> label_source='classifier'
                         conf <  0.7 -> topic_label='noise', topic_id=-1, label_source='review'
  classify_author_new()  youtube_comment -> customer (rule); blog/cafe -> author serve classifier
                         conf >= gate(source, class) -> author_type_src='classifier'
                         else author_type_src='author_review'
  llm_author_review()    author_review docs -> Haiku two-step prompt (analysis.author_llm)
  apply_author_filter()  blog/cafe author_type='unrelated' -> topic_label='noise' (author_filter)
  llm_tiebreak()         category review docs -> Haiku picks a category or 'noise'
  sentiment_new()        docs without sentiment (excluding noise) -> dictionary + Haiku

Embeddings go through analysis.embedding_cache so weekly docs are cached for
the quarterly retrain. Never deletes existing rows; safe to re-run.
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from config.db import get_conn
from config.settings import BRAND_KR_TO_KEY
from etl.staging.text_preprocessing import clean_text
from analysis.brand_rules import resolve_brand

SERVE_DIR = "models/serve"
WEB_SOURCES = ("blog", "cafearticle")


def ensure_columns():
    with get_conn() as conn:
        with conn.cursor() as cur:
            for col, typ in [("brand_query", "VARCHAR(50)"), ("brand_source", "VARCHAR(20)"),
                             ("author_type", "VARCHAR(20)"), ("author_type_src", "VARCHAR(20)"),
                             ("author_conf", "NUMERIC(5,4)")]:
                cur.execute(f"ALTER TABLE staging.content_enriched ADD COLUMN IF NOT EXISTS {col} {typ}")
            cur.execute("ALTER TABLE staging.content_enriched ALTER COLUMN author_type_src TYPE VARCHAR(20)")
            cur.execute("ALTER TABLE staging.content_enriched ALTER COLUMN author_type TYPE VARCHAR(20)")


# ------------------------------------------------------------------
# 1. Stage new raw rows
# ------------------------------------------------------------------

def stage_new():
    """Insert raw docs that are not yet in staging. Returns count."""
    ensure_columns()
    with get_conn() as conn:
        voc = pd.read_sql(
            """
            SELECT v.id, v.source, v.title, v.description, v.brand, v.postdate
            FROM raw.naver_voc v
            LEFT JOIN staging.content_enriched s
              ON s.source = v.source AND s.source_id = v.id::text
            WHERE s.id IS NULL
            """, conn)
        com = pd.read_sql(
            """
            SELECT c.comment_id, c.text_original, c.published_at, yv.brand
            FROM raw.youtube_comments c
            JOIN raw.youtube_videos yv ON c.video_id = yv.video_id
            LEFT JOIN staging.content_enriched s
              ON s.source = 'youtube_comment' AND s.source_id = c.comment_id
            WHERE s.id IS NULL
            """, conn)

    rows = []
    for _, r in voc.iterrows():
        t, d = clean_text(r["title"]), clean_text(r["description"])
        txt = f"{t} {d}".strip()
        if len(txt) >= 10:
            b = BRAND_KR_TO_KEY.get(r["brand"], r["brand"])
            rows.append((r["source"], str(r["id"]), b, b, t, txt, r["postdate"]))
    for _, r in com.iterrows():
        txt = clean_text(r["text_original"])
        if len(txt) >= 10:
            b = BRAND_KR_TO_KEY.get(r["brand"], r["brand"])
            rows.append(("youtube_comment", r["comment_id"], b, b, "", txt, r["published_at"]))

    if rows:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO staging.content_enriched
                        (source, source_id, brand, brand_query, title, text_clean, published_at,
                         label_source, brand_source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending', 'query')
                    """, rows)
    print(f"[stage_new] {len(rows)} new docs staged (voc {len(voc)}, comments {len(com)} raw)")
    return len(rows)


# ------------------------------------------------------------------
# 1b. Title-based brand rule for new web docs
# ------------------------------------------------------------------

def resolve_brands_new():
    """Apply the title brand rule to docs staged this run (brand_source='query', pending)."""
    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT id, brand, COALESCE(title, '') AS title
            FROM staging.content_enriched
            WHERE source = ANY(%s) AND label_source = 'pending'
              AND COALESCE(brand_source, 'query') = 'query'
            """, conn, params=(list(WEB_SOURCES),))
    changes = []
    for _, r in df.iterrows():
        nb, changed = resolve_brand(r["brand"], r["title"])
        if changed:
            changes.append((nb, int(r["id"])))
    if changes:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    "UPDATE staging.content_enriched SET brand=%s, brand_source='title' WHERE id=%s",
                    changes)
    print(f"[resolve_brands_new] {len(df)} web docs checked, {len(changes)} reassigned by title")
    return {"n": int(len(df)), "reassigned": len(changes)}


# ------------------------------------------------------------------
# 2. Classify pending docs (category)
# ------------------------------------------------------------------

def classify_new(threshold=None):
    """Assign category to pending docs with the serve classifier."""
    import joblib
    from analysis.embedding_cache import get_embeddings

    meta = json.load(open(f"{SERVE_DIR}/classifier_meta.json", encoding="utf-8"))
    thr = threshold if threshold is not None else 0.7

    with get_conn() as conn:
        df = pd.read_sql(
            "SELECT id, text_clean FROM staging.content_enriched WHERE label_source = 'pending' ORDER BY id",
            conn)
    if df.empty:
        print("[classify_new] nothing pending")
        return {"n": 0, "confirmed": 0, "review": 0, "review_rate": 0.0}

    X = get_embeddings(df["id"].values, df["text_clean"].tolist())
    clf = joblib.load(f"{SERVE_DIR}/classifier.joblib")
    proba = clf.predict_proba(X)
    pred = clf.classes_[proba.argmax(1)]
    conf = proba.max(1)

    confirmed = conf >= thr
    with get_conn() as conn:
        with conn.cursor() as cur:
            for doc_id, p, c, ok in zip(df["id"], pred, conf, confirmed):
                if ok:
                    cur.execute(
                        "UPDATE staging.content_enriched SET topic_label=%s, topic_id=-2, "
                        "label_conf=%s, label_source='classifier' WHERE id=%s",
                        (p, float(c), int(doc_id)))
                else:
                    cur.execute(
                        "UPDATE staging.content_enriched SET topic_label='noise', topic_id=-1, "
                        "label_conf=%s, label_source='review' WHERE id=%s",
                        (float(c), int(doc_id)))

    stats = {"n": int(len(df)), "confirmed": int(confirmed.sum()),
             "review": int((~confirmed).sum()),
             "review_rate": round(float((~confirmed).mean()), 4),
             "pred_dist": pd.Series(pred[confirmed]).value_counts().to_dict(),
             "classes": meta.get("classes")}
    print(f"[classify_new] {stats['n']} docs: {stats['confirmed']} confirmed, "
          f"{stats['review']} to review ({stats['review_rate']:.1%})")
    return stats


# ------------------------------------------------------------------
# 2b. Author type for new docs
# ------------------------------------------------------------------

def _author_gate(meta, source, pred_class):
    th = meta.get("source_thresholds", {})
    v = th.get(source)
    if isinstance(v, dict):
        return float(v.get(pred_class, meta.get("op_threshold", 0.7)))
    if isinstance(v, (int, float)):
        return float(v)
    return float(meta.get("op_threshold", 0.7))


def classify_author_new():
    """author_type for docs that have none: youtube by rule, blog/cafe by the author classifier."""
    import joblib
    from analysis.embedding_cache import get_embeddings
    from analysis.author_features import lexical_features, source_flag

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE staging.content_enriched SET author_type='customer', author_type_src='rule' "
                "WHERE source='youtube_comment' AND author_type IS NULL")
            n_rule = cur.rowcount
        df = pd.read_sql(
            """
            SELECT id, source, COALESCE(title, '') AS title, text_clean
            FROM staging.content_enriched
            WHERE source = ANY(%s) AND author_type IS NULL AND label_source <> 'pending'
            ORDER BY id
            """, conn, params=(list(WEB_SOURCES),))
    if df.empty:
        print(f"[classify_author_new] youtube rule {n_rule}; no web docs to classify")
        return {"n_rule": int(n_rule), "n": 0, "confirmed": 0, "review": 0}

    meta = json.load(open(f"{SERVE_DIR}/author_classifier_meta.json", encoding="utf-8"))
    bundle = joblib.load(f"{SERVE_DIR}/author_classifier.joblib")
    model = bundle["model"]

    X = get_embeddings(df["id"].values, df["text_clean"].tolist())
    parts = [X]
    if bundle.get("uses_lex"):
        parts.append(bundle["scaler"].transform(lexical_features(df["title"].tolist(), df["text_clean"].tolist())))
    if bundle.get("uses_src"):
        parts.append(source_flag(df["source"].tolist()))
    Xf = np.hstack(parts) if len(parts) > 1 else X

    proba = model.predict_proba(Xf)
    pred = model.classes_[proba.argmax(1)]
    conf = proba.max(1)
    ok = np.array([c >= _author_gate(meta, s, p) for s, p, c in zip(df["source"], pred, conf)])

    with get_conn() as conn:
        with conn.cursor() as cur:
            for doc_id, s, p, c, k in zip(df["id"], df["source"], pred, conf, ok):
                if k:
                    cur.execute(
                        "UPDATE staging.content_enriched SET author_type=%s, author_type_src='classifier', "
                        "author_conf=%s WHERE id=%s", (p, float(c), int(doc_id)))
                else:
                    cur.execute(
                        "UPDATE staging.content_enriched SET author_type_src='author_review', "
                        "author_conf=%s WHERE id=%s", (float(c), int(doc_id)))

    by_src = {s: {"n": int((df["source"] == s).sum()),
                  "confirmed": int(ok[(df["source"] == s).values].sum())} for s in df["source"].unique()}
    stats = {"n_rule": int(n_rule), "n": int(len(df)), "confirmed": int(ok.sum()),
             "review": int((~ok).sum()), "pred_dist": pd.Series(pred[ok]).value_counts().to_dict(),
             "by_source": by_src}
    print(f"[classify_author_new] youtube rule {n_rule}; web {stats['n']}: "
          f"{stats['confirmed']} confirmed, {stats['review']} to LLM review | {by_src}")
    return stats


def llm_author_review():
    """Label author_review docs with the Haiku two-step prompt (same prompt and parser as the
    backfill), then re-check blog 'customer' answers with Sonnet, exactly like the backfill did."""
    from analysis.author_llm import label_author_docs, verify_customer_docs

    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT id, source, COALESCE(title, '') AS title, text_clean
            FROM staging.content_enriched
            WHERE author_type IS NULL AND author_type_src = 'author_review'
            ORDER BY id
            """, conn)
    if df.empty:
        print("[llm_author_review] nothing to review")
        return {"n": 0}
    labels = label_author_docs(df)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE staging.content_enriched SET author_type=%s, author_type_src='haiku' WHERE id=%s",
                [(lab, int(i)) for i, lab in labels.items()])

    blog_customer = df[(df["source"] == "blog") & df["id"].map(lambda i: labels.get(int(i)) == "customer")]
    verified, changed = {}, 0
    if not blog_customer.empty:
        verified = verify_customer_docs(blog_customer)
        changed = sum(1 for v in verified.values() if v != "customer")
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    "UPDATE staging.content_enriched SET author_type=%s, author_type_src='sonnet' WHERE id=%s",
                    [(lab, int(i)) for i, lab in verified.items()])
    stats = {"n": int(len(df)), "labeled": len(labels),
             "dist": pd.Series(list(labels.values())).value_counts().to_dict() if labels else {},
             "blog_customer_verified": len(verified), "verify_changed": changed}
    print(f"[llm_author_review] {stats['n']} docs: {stats['labeled']} labeled {stats['dist']} | "
          f"blog customer verified {len(verified)}, changed {changed}")
    return stats


def apply_author_filter(max_affected=None):
    """blog/cafe docs whose author is unrelated to insurance are noise for every downstream module.
    Only docs that are not noise yet are touched: docs already noise keep their topic_id and
    label_source (the category classifier trains its noise class on them).
    max_affected: safety stop — if more docs than this would change, nothing is written."""
    cond = "source = ANY(%s) AND author_type = 'unrelated' AND topic_label <> 'noise'"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM staging.content_enriched WHERE {cond}", (list(WEB_SOURCES),))
            n_cand = cur.fetchone()[0]
            if max_affected is not None and n_cand > max_affected:
                print(f"[apply_author_filter] STOP: {n_cand} docs would change, limit {max_affected}")
                return {"n": 0, "candidates": int(n_cand), "stopped": True}
            cur.execute(
                f"""
                UPDATE staging.content_enriched
                SET topic_label='noise', topic_id=-1, label_source='author_filter'
                WHERE {cond}
                """, (list(WEB_SOURCES),))
            n = cur.rowcount
    print(f"[apply_author_filter] {n} web docs set to noise (author unrelated)")
    return {"n": int(n), "candidates": int(n_cand), "stopped": False}


# ------------------------------------------------------------------
# 3. Sentiment for new docs
# ------------------------------------------------------------------

def sentiment_new(threshold=0.3, use_llm=True):
    """Run hybrid sentiment on docs that have no sentiment yet (excluding review/noise)."""
    from analysis.sentiment import dict_sentiment, score_to_label, call_haiku_batch

    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT id, text_clean FROM staging.content_enriched
            WHERE sentiment_label IS NULL AND topic_label != 'noise'
            ORDER BY id
            """, conn)
    if df.empty:
        print("[sentiment_new] nothing to score")
        return {"n": 0}

    scores, confs, labels, methods = [], [], [], []
    for t in df["text_clean"]:
        s, c, _ = dict_sentiment(t)
        scores.append(s); confs.append(c); labels.append(score_to_label(s, c))
        methods.append("dictionary" if c >= threshold else ("low_conf_dict" if c == 0 else "llm"))
    df["score"], df["conf"], df["label"], df["method"] = scores, confs, labels, methods

    llm_idx = df.index[df["method"] == "llm"].tolist()
    if use_llm and llm_idx:
        res = call_haiku_batch(df.loc[llm_idx, "text_clean"].tolist(), batch_size=20)
        for i, (s, l) in zip(llm_idx, res):
            df.loc[i, "score"], df.loc[i, "label"] = s, l
    elif llm_idx:
        df.loc[llm_idx, "method"] = "low_conf_dict"

    with get_conn() as conn:
        with conn.cursor() as cur:
            for _, r in df.iterrows():
                cur.execute(
                    "UPDATE staging.content_enriched SET sentiment_score=%s, sentiment_label=%s, "
                    "sentiment_method=%s WHERE id=%s",
                    (float(r["score"]), r["label"], r["method"], int(r["id"])))

    stats = {"n": int(len(df)), "methods": df["method"].value_counts().to_dict(),
             "labels": df["label"].value_counts().to_dict()}
    print(f"[sentiment_new] {stats['n']} docs scored: {stats['methods']}")
    return stats


# ------------------------------------------------------------------
# 4. Category tie-break for review docs
# ------------------------------------------------------------------

def llm_tiebreak(batch_size=20):
    """Second stage for review docs: Haiku picks a category or 'noise'.
    Runs after apply_author_filter so unrelated-author docs never reach it."""
    import os, re, json as _json
    import anthropic
    cats = ["자동차보험", "실손·건강보험", "보험금 청구·보상", "상품 설계·고객후기", "태아·어린이보험",
            "브랜드 캠페인", "이벤트·프로모션", "noise"]
    with get_conn() as conn:
        df = pd.read_sql("SELECT id, text_clean FROM staging.content_enriched "
                         "WHERE label_source='review' ORDER BY id", conn)
    if df.empty:
        print("[llm_tiebreak] nothing to tie-break"); return {"n": 0}
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    model = os.getenv("LLM_MODEL_SENTIMENT", "claude-haiku-4-5-20251001")
    out = {}
    for i in range(0, len(df), batch_size):
        b = df.iloc[i:i+batch_size]
        block = "\n".join(f"[{j+1}] {t[:300]}" for j, t in enumerate(b["text_clean"]))
        prompt = (f"손해보험 VoC 분류. 각 텍스트를 다음 중 하나로: {', '.join(cats)}.\n"
                  "보험과 무관하거나 광고·채용·업무용이면 noise. JSON 배열만 출력: "
                  '[{"id":1,"category":"..."}]\n\n' + block)
        try:
            r = client.messages.create(model=model, max_tokens=800,
                                       messages=[{"role": "user", "content": prompt}])
            txt = re.sub(r"```json|```", "", r.content[0].text).strip()
            m = re.search(r"\[.*\]", txt, flags=re.S)
            for item in _json.loads(m.group(0) if m else txt):
                k = int(item["id"]) - 1
                if 0 <= k < len(b) and item.get("category") in cats:
                    out[int(b.iloc[k]["id"])] = item["category"]
        except Exception as e:
            print(f"  [WARN] tiebreak batch error: {e}")
    with get_conn() as conn:
        with conn.cursor() as cur:
            for doc_id, cat in out.items():
                if cat == "noise":
                    cur.execute("UPDATE staging.content_enriched SET label_source='review_noise' WHERE id=%s", (doc_id,))
                else:
                    cur.execute("UPDATE staging.content_enriched SET topic_label=%s, topic_id=-2, "
                                "label_source='llm' WHERE id=%s", (cat, doc_id))
    rescued = sum(1 for c in out.values() if c != "noise")
    print(f"[llm_tiebreak] {len(df)} review docs: {rescued} rescued, {len(out)-rescued} confirmed noise")
    return {"n": len(df), "rescued": rescued}
