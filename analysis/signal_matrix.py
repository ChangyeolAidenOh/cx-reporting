"""Analysis: Content Signal Matrix, brand x category significance, lead-lag, representatives.

Part 1  Engagement vs Search Interest matrix per brand x topic category
Part 2  Brand x category proportion test (chi-square + Bonferroni-corrected
        post-hoc z-tests) -> which brands over/under-index on which categories
Part 3  Lead-lag cross-correlation (MSTL residual) content volume vs search
Part 4  Representative documents per brand x category (for report quotes)

Usage:
    python -m analysis.signal_matrix
"""

import sys
import os
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from config.db import get_conn
from config.settings import BRANDS, BRAND_KR_TO_KEY, TARGET_BRAND

EXPORT_DIR = "data/exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

# Signal strength tags (adopted from prior VoC brief methodology)
TAG_STAT = "통계 검증 ✓"
TAG_CLUSTER = "cluster 시그널"
TAG_QUAL = "정성 시그널"


# ============================================================
# Part 1: Content Signal Matrix
# ============================================================

def build_signal_matrix():
    print("=" * 70)
    print("Part 1: Content Signal Matrix")
    print("=" * 70)

    with get_conn() as conn:
        content_df = pd.read_sql(
            """
            SELECT s.brand, s.topic_label, s.published_at,
                   v.view_count, v.like_count, v.comment_count
            FROM staging.content_enriched s
            LEFT JOIN raw.youtube_comments c ON s.source_id = c.comment_id
            LEFT JOIN raw.youtube_videos v ON c.video_id = v.video_id
            WHERE s.topic_label != 'noise'
              AND s.source = 'youtube_comment'
              AND s.published_at IS NOT NULL
            """, conn)
        search_df = pd.read_sql(
            "SELECT brand, week_start, search_ratio, wow_change_pct FROM mart.brand_search_weekly",
            conn)

    content_df["published_at"] = pd.to_datetime(content_df["published_at"])
    search_df["week_start"] = pd.to_datetime(search_df["week_start"])
    search_df["brand_key"] = search_df["brand"].map(BRAND_KR_TO_KEY)
    content_df["week_start"] = content_df["published_at"].dt.to_period("W-SUN").apply(lambda x: x.start_time)
    content_df["eng"] = content_df["like_count"].fillna(0) + content_df["comment_count"].fillna(0)

    engagement = content_df.groupby(["brand", "topic_label", "week_start"]).agg(
        total_engagement=("eng", "sum"), content_count=("brand", "count")).reset_index()

    records = []
    for (brand, topic), g in engagement.groupby(["brand", "topic_label"]):
        avg_eng = g["total_engagement"].mean()
        signals = []
        for _, r in g.iterrows():
            nxt = r["week_start"] + pd.Timedelta(weeks=1)
            m = search_df[(search_df["brand_key"] == brand) & (search_df["week_start"] == nxt)]
            if not m.empty and pd.notna(m.iloc[0]["wow_change_pct"]):
                signals.append(m.iloc[0]["wow_change_pct"])
        sig = float(np.mean(signals)) if signals else 0.0
        records.append({
            "brand": brand, "topic_label": topic,
            "avg_engagement": round(avg_eng, 2), "search_signal": round(sig, 4),
            "quadrant": classify_quadrant(avg_eng, sig), "sample_count": len(g),
            "signal_tag": TAG_CLUSTER,
        })
    matrix_df = pd.DataFrame(records)

    print(f"\n{'Brand':<14} {'Category':<18} {'Engagement':>11} {'Signal':>9} {'Quadrant':<10} N")
    print("-" * 72)
    for _, r in matrix_df.sort_values(["brand", "quadrant"]).iterrows():
        print(f"{r['brand']:<14} {r['topic_label']:<18} {r['avg_engagement']:>11.1f} "
              f"{r['search_signal']:>9.4f} {r['quadrant']:<10} {r['sample_count']}")
    return matrix_df


def classify_quadrant(engagement, search_signal):
    high_eng = engagement > 100
    high_sig = search_signal > 0
    if high_eng and high_sig:
        return "high-high"
    if high_eng:
        return "high-low"
    if high_sig:
        return "low-high"
    return "low-low"


def save_matrix_to_db(matrix_df):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM mart.content_signal_matrix")
            for _, r in matrix_df.iterrows():
                cur.execute(
                    """
                    INSERT INTO mart.content_signal_matrix
                        (brand, topic_label, avg_engagement, search_signal, quadrant, sample_count)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (r["brand"], r["topic_label"], float(r["avg_engagement"]),
                     float(r["search_signal"]), r["quadrant"], int(r["sample_count"])))
    print(f"\nSaved {len(matrix_df)} rows to mart.content_signal_matrix")


# ============================================================
# Part 2: Brand x Category proportion test
# ============================================================

def brand_category_test(alpha=0.05):
    """Chi-square on brand x category table; Bonferroni-corrected post-hoc
    z-tests of each (brand, category) proportion vs. all other brands."""
    print("\n" + "=" * 70)
    print("Part 2: Brand x Category Proportion Test (chi-square + Bonferroni)")
    print("=" * 70)

    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT s.brand, s.topic_label
            FROM staging.content_enriched s
            LEFT JOIN raw.naver_voc v
              ON s.source IN ('blog', 'cafearticle') AND s.source_id = v.id::text
            WHERE s.topic_label != 'noise' AND s.brand != 'general'
              AND (v.query IS NULL OR v.query LIKE '%%후기' OR v.query IN ('자동차보험 비교', '실손보험 추천'))
            """, conn)

    ct = pd.crosstab(df["brand"], df["topic_label"])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    print("  (symmetric subset: target-brand product queries excluded from cross-brand test)")
    print(f"\nOmnibus chi-square: chi2={chi2:.1f}, dof={dof}, p={p:.2e}")

    n_tests = ct.shape[0] * ct.shape[1]
    alpha_bonf = alpha / n_tests
    print(f"Post-hoc: {n_tests} cells, Bonferroni alpha = {alpha_bonf:.5f}")

    rows = []
    for brand in ct.index:
        n_b = ct.loc[brand].sum()
        n_o = ct.drop(index=brand).values.sum()
        for cat in ct.columns:
            k_b = ct.loc[brand, cat]
            k_o = ct.drop(index=brand)[cat].sum()
            p_b, p_o = k_b / n_b, k_o / n_o
            p_pool = (k_b + k_o) / (n_b + n_o)
            se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_b + 1 / n_o))
            z = (p_b - p_o) / se if se > 0 else 0.0
            pval = 2 * (1 - stats.norm.cdf(abs(z)))
            ratio = p_b / p_o if p_o > 0 else np.nan
            rows.append({
                "brand": brand, "topic_label": cat,
                "share_brand": round(p_b, 4), "share_others": round(p_o, 4),
                "ratio": round(ratio, 2), "z": round(z, 2), "p_value": pval,
                "significant": pval < alpha_bonf,
                "signal_tag": TAG_STAT if pval < alpha_bonf else TAG_CLUSTER,
            })
    res = pd.DataFrame(rows)

    print(f"\nSignificant over/under-indexing (Bonferroni, alpha={alpha}):")
    sig = res[res["significant"]].sort_values(["brand", "ratio"], ascending=[True, False])
    for _, r in sig.iterrows():
        direction = "over" if r["ratio"] > 1 else "under"
        mark = " <- TARGET" if r["brand"] == TARGET_BRAND else ""
        print(f"  {r['brand']:<14} {r['topic_label']:<18} {r['ratio']:>5.2f}x  "
              f"({r['share_brand']:.1%} vs {r['share_others']:.1%})  z={r['z']:+.1f}  {direction}{mark}")

    res.to_csv(f"{EXPORT_DIR}/brand_category_test.csv", index=False, encoding="utf-8-sig")
    print(f"\nCSV exported: {EXPORT_DIR}/brand_category_test.csv")
    return res


# ============================================================
# Part 3: Lead-Lag
# ============================================================

def lead_lag_analysis():
    print("\n" + "=" * 70)
    print("Part 3: Lead-Lag Analysis (MSTL residual cross-correlation)")
    print("=" * 70)
    from statsmodels.tsa.seasonal import MSTL

    with get_conn() as conn:
        content_weekly = pd.read_sql(
            """
            SELECT brand, DATE_TRUNC('week', published_at::timestamp)::date AS week_start,
                   COUNT(*) AS content_volume
            FROM staging.content_enriched
            WHERE topic_label != 'noise' AND published_at IS NOT NULL
            GROUP BY brand, DATE_TRUNC('week', published_at::timestamp)::date
            """, conn)
        search_weekly = pd.read_sql(
            "SELECT brand, week_start, search_ratio FROM mart.brand_search_weekly", conn)

    content_weekly["week_start"] = pd.to_datetime(content_weekly["week_start"])
    search_weekly["week_start"] = pd.to_datetime(search_weekly["week_start"])
    search_weekly["brand_key"] = search_weekly["brand"].map(BRAND_KR_TO_KEY)

    results = {}
    for brand_key in BRANDS:
        cdf = content_weekly[content_weekly["brand"] == brand_key].set_index("week_start").sort_index()
        sdf = search_weekly[search_weekly["brand_key"] == brand_key].set_index("week_start").sort_index()
        idx = cdf.index.intersection(sdf.index)
        if len(idx) < 10:
            print(f"\n  {brand_key}: insufficient overlap ({len(idx)} weeks)")
            continue
        cs = cdf.loc[idx, "content_volume"].astype(float)
        ss = sdf.loc[idx, "search_ratio"].astype(float)
        try:
            s_res = MSTL(ss, periods=[4, 13]).fit().resid if len(ss) >= 14 else ss - ss.mean()
        except Exception:
            s_res = ss - ss.mean()
        c_res = cs - cs.rolling(4, min_periods=1).mean()

        max_lag = min(8, len(idx) // 3)
        lags = {}
        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                r = c_res.corr(s_res)
            elif lag > 0:
                r = c_res.iloc[:-lag].reset_index(drop=True).corr(s_res.iloc[lag:].reset_index(drop=True))
            else:
                r = c_res.iloc[-lag:].reset_index(drop=True).corr(s_res.iloc[:lag].reset_index(drop=True))
            if pd.notna(r):
                lags[lag] = round(float(r), 4)
        results[brand_key] = lags
        best = max(lags, key=lambda k: abs(lags[k]))
        tag = " <- TARGET" if brand_key == TARGET_BRAND else ""
        print(f"  {brand_key:<14} n={len(idx):>3}w  best lag={best:+d}  r={lags[best]:+.4f}{tag}")

    with open(f"{EXPORT_DIR}/lead_lag_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nJSON exported: {EXPORT_DIR}/lead_lag_results.json")
    return results


# ============================================================
# Part 4: Representatives
# ============================================================

def extract_representatives(n_per_cell=3, min_len=40, max_len=220):
    """Pick representative docs per brand x category x sentiment for report quotes.
    Preference: mid-length, high |sentiment_score|, from the target brand first."""
    print("\n" + "=" * 70)
    print("Part 4: Representative documents")
    print("=" * 70)

    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT id, brand, topic_label, sentiment_label, sentiment_score,
                   source, text_clean, published_at
            FROM staging.content_enriched
            WHERE topic_label != 'noise' AND sentiment_label IS NOT NULL
              AND LENGTH(text_clean) BETWEEN %s AND %s
            """, conn, params=(min_len, max_len))

    ad_markers = r"변호사|법무법인|무료상담|손해사정사|연락처|문의주세요|상담신청|바로가기|기자|뉴스|보도|\d{2,3}-\d{3,4}-\d{4}"
    df = df[~df["text_clean"].str.contains(ad_markers, regex=True, na=False)]

    df["abs_score"] = df["sentiment_score"].abs()
    src_rank = {"youtube_comment": 0, "cafearticle": 1, "blog": 2}
    df["src_rank"] = df["source"].map(src_rank).fillna(3)
    reps = (df.sort_values(["src_rank", "abs_score"], ascending=[True, False])
            .groupby(["brand", "topic_label", "sentiment_label"])
            .head(n_per_cell)
            .sort_values(["brand", "topic_label", "sentiment_label"]))

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS mart.topic_representatives (
                    id SERIAL PRIMARY KEY,
                    doc_id INTEGER, brand VARCHAR(50), topic_label VARCHAR(50),
                    sentiment_label VARCHAR(20), sentiment_score NUMERIC(5,3),
                    source VARCHAR(30), text_clean TEXT, published_at TIMESTAMP
                )""")
            cur.execute("DELETE FROM mart.topic_representatives")
            for _, r in reps.iterrows():
                cur.execute(
                    """
                    INSERT INTO mart.topic_representatives
                        (doc_id, brand, topic_label, sentiment_label, sentiment_score,
                         source, text_clean, published_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (int(r["id"]), r["brand"], r["topic_label"], r["sentiment_label"],
                     float(r["sentiment_score"]), r["source"], r["text_clean"],
                     r["published_at"] if pd.notna(r["published_at"]) else None))
    reps.drop(columns=["abs_score", "src_rank"]).to_csv(f"{EXPORT_DIR}/topic_representatives.csv",
                                          index=False, encoding="utf-8-sig")
    print(f"Saved {len(reps)} representatives to mart.topic_representatives")

    tgt = reps[(reps["brand"] == TARGET_BRAND) & (reps["sentiment_label"] == "negative")]
    print(f"\nTarget brand negative representatives (sample):")
    for _, r in tgt.head(5).iterrows():
        print(f"  [{r['topic_label']}] {r['text_clean'][:90]}")
    return reps


# ============================================================
# Main
# ============================================================

def run():
    matrix_df = build_signal_matrix()
    save_matrix_to_db(matrix_df)
    matrix_df.to_csv(f"{EXPORT_DIR}/content_signal_matrix.csv", index=False, encoding="utf-8-sig")

    brand_category_test()
    lead_lag_analysis()
    extract_representatives()

    print("\n" + "=" * 70)
    print("Signal analysis complete.")
    print("  mart.content_signal_matrix, mart.topic_representatives")
    print(f"  {EXPORT_DIR}/content_signal_matrix.csv")
    print(f"  {EXPORT_DIR}/brand_category_test.csv")
    print(f"  {EXPORT_DIR}/lead_lag_results.json")
    print(f"  {EXPORT_DIR}/topic_representatives.csv")


if __name__ == "__main__":
    run()
