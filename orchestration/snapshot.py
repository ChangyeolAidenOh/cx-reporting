"""Weekly snapshot of key metrics and week-over-week diff.

snapshot_json keys:
  sov:           {brand_key: {sov, wow, rank}}
  cat_index:     {category: {ratio, significant}}   (target brand, symmetric subset)
  sentiment:     {positive, neutral, negative, n}   (target brand, 4-week rolling)
  anomalies:     [{brand, date, z, methods}]        (last 4 weeks)
  pipeline:      {new_docs, review_rate, outlier_rate_total, ...}
  representatives: {category: [doc_id,...]}         (target brand negatives)

diff_json: computed against the previous snapshot row.
"""

import sys
import os
import json
from datetime import timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from config.db import get_conn
from config.settings import BRANDS, BRAND_KR_TO_KEY, TARGET_BRAND


def build_snapshot(week_start, pipeline_stats):
    week_label = f"{week_start.year}-W{week_start.isocalendar()[1]:02d}"
    win_start = week_start - timedelta(days=28)

    with get_conn() as conn:
        sov = pd.read_sql(
            "SELECT brand, sov_pct, wow_change_pct FROM mart.brand_search_weekly WHERE week_start = %s "
            "ORDER BY sov_pct DESC", conn, params=(week_start,))
        sent = pd.read_sql(
            "SELECT AVG(positive_ratio) p, AVG(neutral_ratio) n, AVG(negative_ratio) ng, SUM(total_docs) d "
            "FROM mart.voc_sentiment WHERE brand = %s AND week_start BETWEEN %s AND %s",
            conn, params=(TARGET_BRAND, win_start, week_start))
        anom = pd.read_sql(
            "SELECT brand, detected_date, zscore, methods_agreed FROM mart.anomaly_log "
            "WHERE detected_date BETWEEN %s AND %s ORDER BY detected_date",
            conn, params=(win_start, week_start))
        reps = pd.read_sql(
            "SELECT topic_label, doc_id FROM mart.topic_representatives "
            "WHERE brand = %s AND sentiment_label = 'negative'", conn, params=(TARGET_BRAND,))
        totals = pd.read_sql(
            "SELECT COUNT(*) total, SUM((topic_id = -1)::int) outliers, "
            "SUM((label_source='review')::int) review FROM staging.content_enriched", conn)

    cat_path = "data/exports/brand_category_test.csv"
    cat = pd.read_csv(cat_path) if os.path.exists(cat_path) else pd.DataFrame()
    cat_t = cat[cat["brand"] == TARGET_BRAND] if not cat.empty else pd.DataFrame()

    snap = {
        "week_label": week_label,
        "sov": {BRAND_KR_TO_KEY.get(r["brand"], r["brand"]): {
                    "sov": round(float(r["sov_pct"]), 2),
                    "wow": None if pd.isna(r["wow_change_pct"]) else round(float(r["wow_change_pct"]), 2),
                    "rank": i + 1}
                for i, (_, r) in enumerate(sov.iterrows())},
        "cat_index": {r["topic_label"]: {"ratio": round(float(r["ratio"]), 3),
                                         "significant": bool(r["significant"])}
                      for _, r in cat_t.iterrows()},
        "sentiment": {"positive": round(float(sent.iloc[0]["p"] or 0), 4),
                      "neutral": round(float(sent.iloc[0]["n"] or 0), 4),
                      "negative": round(float(sent.iloc[0]["ng"] or 0), 4),
                      "n": int(sent.iloc[0]["d"] or 0)},
        "anomalies": [{"brand": BRAND_KR_TO_KEY.get(r["brand"], r["brand"]),
                       "date": str(r["detected_date"]), "z": round(float(r["zscore"]), 2),
                       "methods": int(r["methods_agreed"])} for _, r in anom.iterrows()],
        "representatives": reps.groupby("topic_label")["doc_id"].apply(list).to_dict(),
        "pipeline": {**pipeline_stats,
                     "total_docs": int(totals.iloc[0]["total"]),
                     "outlier_rate_total": round(float(totals.iloc[0]["outliers"] / totals.iloc[0]["total"]), 4),
                     "review_queue": int(totals.iloc[0]["review"] or 0)},
    }
    return week_label, snap


def diff_snapshots(prev, cur):
    """Compute what changed. Returns dict of change lists + alerts."""
    if not prev:
        return {"first_snapshot": True, "alerts": []}
    d = {"alerts": [], "sov": [], "cat_index": [], "sentiment": {}, "new_anomalies": [], "representatives": {}}

    # SoV / rank
    for b, c in cur["sov"].items():
        p = prev["sov"].get(b)
        if not p:
            continue
        if p["rank"] != c["rank"]:
            d["sov"].append({"brand": b, "rank": [p["rank"], c["rank"]], "sov": [p["sov"], c["sov"]]})
        elif abs(c["sov"] - p["sov"]) >= 1.0:
            d["sov"].append({"brand": b, "sov": [p["sov"], c["sov"]]})

    # Category index: significance flips or ratio change >= 20%
    for k, c in cur["cat_index"].items():
        p = prev["cat_index"].get(k)
        if not p:
            continue
        flip = p["significant"] != c["significant"]
        rel = (c["ratio"] - p["ratio"]) / p["ratio"] if p["ratio"] else 0
        if flip or abs(rel) >= 0.20:
            d["cat_index"].append({"category": k, "ratio": [p["ratio"], c["ratio"]],
                                   "significant": [p["significant"], c["significant"]], "flip": flip})
            if flip:
                d["alerts"].append(f"유의성 전환: {k} ({p['ratio']:.2f}x→{c['ratio']:.2f}x) — 큐레이션 인사이트 재검토")

    # Sentiment (pp change)
    for k in ["positive", "negative"]:
        delta = cur["sentiment"][k] - prev["sentiment"][k]
        d["sentiment"][k] = round(delta * 100, 2)
        if k == "negative" and delta >= 0.03:
            d["alerts"].append(f"부정 비율 +{delta*100:.1f}pp — 원인 조사 필요")

    # New anomalies
    prev_keys = {(a["brand"], a["date"]) for a in prev["anomalies"]}
    d["new_anomalies"] = [a for a in cur["anomalies"] if (a["brand"], a["date"]) not in prev_keys]
    for a in d["new_anomalies"]:
        if a["methods"] == 3 and a["brand"] == TARGET_BRAND:
            d["alerts"].append(f"3/3 이상 이벤트 {a['date']} (z={a['z']:+.2f}) — probable_cause 기입 필요")

    # Representatives turnover
    for k, ids in cur["representatives"].items():
        pids = set(prev["representatives"].get(k, []))
        new_ids = [i for i in ids if i not in pids]
        if new_ids:
            d["representatives"][k] = new_ids

    # Pipeline health
    pl = cur["pipeline"]
    if pl.get("review_rate", 0) >= 0.30:
        d["alerts"].append(f"신규 문서 검토 비율 {pl['review_rate']:.0%} ≥ 30% — 모델 드리프트 의심, 재학습 검토")
    if pl.get("llm_share", 0) >= 0.40:
        d["alerts"].append(f"Haiku 호출 비율 {pl['llm_share']:.0%} ≥ 40% — 감성사전 보강 필요")
    return d


def save_snapshot(week_label, week_start, snap, diff):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO mart.weekly_snapshot (week_label, week_start, snapshot_json, diff_json)
                VALUES (%s, %s, %s::jsonb, %s::jsonb)
                ON CONFLICT (week_label) DO UPDATE SET snapshot_json = EXCLUDED.snapshot_json,
                    diff_json = EXCLUDED.diff_json, created_at = NOW()
                """, (week_label, week_start, json.dumps(snap, ensure_ascii=False, default=str),
                      json.dumps(diff, ensure_ascii=False, default=str)))
    os.makedirs("data/exports", exist_ok=True)
    json.dump({"week_label": week_label, "snapshot": snap, "diff": diff},
              open("data/exports/what_changed.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
    print(f"[snapshot] saved {week_label}; alerts: {len(diff.get('alerts', []))}")


def load_prev_snapshot(week_label):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT snapshot_json FROM mart.weekly_snapshot WHERE week_label < %s "
                        "ORDER BY week_start DESC LIMIT 1", (week_label,))
            r = cur.fetchone()
    return r[0] if r else None


def run(week_start, pipeline_stats):
    week_label, snap = build_snapshot(week_start, pipeline_stats)
    prev = load_prev_snapshot(week_label)
    diff = diff_snapshots(prev, snap)
    save_snapshot(week_label, week_start, snap, diff)
    return snap, diff
