"""Incremental steps for the weekly pipeline.

- stage_new():     raw rows not yet in staging -> staging (label_source='pending')
- classify_new():  pending docs -> serve classifier -> category + confidence
                   conf >= threshold -> label_source='classifier'
                   conf <  threshold -> topic_label='noise', topic_id=-1, label_source='review'
- sentiment_new(): pending-sentiment docs -> dictionary + Haiku (reuses analysis.sentiment)

Never deletes existing rows; safe to re-run.
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

SERVE_DIR = "models/serve"


# ------------------------------------------------------------------
# 1. Stage new raw rows
# ------------------------------------------------------------------

def stage_new():
    """Insert raw docs that are not yet in staging. Returns count."""
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
            rows.append((r["source"], str(r["id"]), BRAND_KR_TO_KEY.get(r["brand"], r["brand"]),
                         t, txt, r["postdate"]))
    for _, r in com.iterrows():
        txt = clean_text(r["text_original"])
        if len(txt) >= 10:
            rows.append(("youtube_comment", r["comment_id"], BRAND_KR_TO_KEY.get(r["brand"], r["brand"]),
                         "", txt, r["published_at"]))

    if rows:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO staging.content_enriched
                        (source, source_id, brand, title, text_clean, published_at, label_source)
                    VALUES (%s, %s, %s, %s, %s, %s, 'pending')
                    """, rows)
    print(f"[stage_new] {len(rows)} new docs staged (voc {len(voc)}, comments {len(com)} raw)")
    return len(rows)


# ------------------------------------------------------------------
# 2. Classify pending docs
# ------------------------------------------------------------------

def classify_new(threshold=None):
    """Assign category to pending docs with the serve classifier."""
    import joblib
    from sentence_transformers import SentenceTransformer

    meta = json.load(open(f"{SERVE_DIR}/classifier_meta.json", encoding="utf-8"))
    thr = threshold if threshold is not None else 0.7

    with get_conn() as conn:
        df = pd.read_sql(
            "SELECT id, text_clean FROM staging.content_enriched WHERE label_source = 'pending' ORDER BY id",
            conn)
    if df.empty:
        print("[classify_new] nothing pending")
        return {"n": 0, "confirmed": 0, "review": 0, "review_rate": 0.0}

    emb = SentenceTransformer(meta["embedding_model"])
    X = emb.encode(df["text_clean"].tolist(), batch_size=64, normalize_embeddings=True,
                   show_progress_bar=False).astype(np.float32)
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
             "pred_dist": pd.Series(pred[confirmed]).value_counts().to_dict()}
    print(f"[classify_new] {stats['n']} docs: {stats['confirmed']} confirmed, "
          f"{stats['review']} to review ({stats['review_rate']:.1%})")
    return stats


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

def llm_tiebreak(batch_size=20):
    """Second stage for review docs: Haiku picks a category or 'noise'."""
    import os, re, json as _json
    import anthropic
    from config.settings import BRANDS
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
            for item in _json.loads(txt):
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
