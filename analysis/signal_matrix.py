"""Analysis: Content Signal Matrix, brand x category significance, lead-lag, representatives.

Part 1  Engagement vs Search Interest matrix per brand x topic category
Part 2  Brand x category proportion test (chi-square + Bonferroni-corrected
        post-hoc z-tests) -> which brands over/under-index on which categories,
        with per-cell minimum detectable effect (MDE) and a leave-one-out
        robustness check so that a cell driven by one dominant competitor in
        the "others" pool is not reported as a brand characteristic
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

def _two_proportion_z(k_b, n_b, k_o, n_o):
    """Pooled two-proportion z-test. Returns (ratio, z, p)."""
    p_b, p_o = k_b / n_b, k_o / n_o
    p_pool = (k_b + k_o) / (n_b + n_o)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_b + 1 / n_o))
    z = (p_b - p_o) / se if se > 0 else 0.0
    pval = 2 * (1 - stats.norm.cdf(abs(z)))
    ratio = p_b / p_o if p_o > 0 else np.nan
    return ratio, z, pval


def _cell_test(k_b, n_b, k_o, n_o, small):
    """Cell p-value: z-test for large cells, Fisher exact when the brand cell
    is small (normal approximation is not reliable there). Returns (ratio, z, p, method)."""
    ratio, z, p_z = _two_proportion_z(k_b, n_b, k_o, n_o)
    if small:
        _, p_f = stats.fisher_exact([[k_b, n_b - k_b], [k_o, n_o - k_o]])
        return ratio, z, p_f, "fisher"
    return ratio, z, p_z, "z"


def _shrunk_ratio(k_b, n_b, p_o, m=20, level=0.95):
    """Empirical-Bayes shrinkage of the brand share toward the others' share.
    Prior Beta(m*p_o, m*(1-p_o)) = m pseudo-docs at the pooled composition, so a
    1-of-20 cell cannot print as 133x. Returns (ratio_shrunk, ci_low, ci_high)."""
    if not (0 < p_o < 1):
        return np.nan, np.nan, np.nan
    a = k_b + m * p_o
    b = (n_b - k_b) + m * (1 - p_o)
    lo, hi = stats.beta.ppf([(1 - level) / 2, 1 - (1 - level) / 2], a, b)
    return (a / (a + b)) / p_o, lo / p_o, hi / p_o


def _n_needed(p_o, n_b, n_o, z_total, target_ratio):
    """Brand docs needed to detect target_ratio at the current others' share,
    keeping the current n_others/n_brand proportion."""
    delta = abs(target_ratio - 1) * p_o
    if delta <= 0 or not (0 < p_o < 1) or n_b == 0:
        return np.nan
    k = n_o / n_b
    return int(np.ceil((1 + 1 / k) * p_o * (1 - p_o) * z_total ** 2 / delta ** 2))


def brand_category_test(alpha=0.05, author_types=None, sources=None, label="all",
                        min_cell_n=5, min_col_n=30, small_cell_n=30, shrink_m=20, power=0.8):
    """Chi-square + Bonferroni post-hoc + per-cell MDE + leave-one-out robustness.

    author_types / sources: None (all) or list to restrict the segment.
    min_col_n: categories with fewer docs than this in the segment are excluded
        from the test (expected-count assumption of chi-square is not met).
    min_cell_n: cells with fewer docs than this get ratio/z/p = NaN and are
        never reported as significant.
    MDE: smallest brand/others ratio that would reach the Bonferroni threshold
        with the given power at the observed n_brand, n_others and p_others.
        A non-significant cell means "no difference beyond
        [mde_ratio_under, mde_ratio_over]", not "no difference".
    Small cells (n_brand_cat < small_cell_n): p-value from Fisher exact instead of
        the z approximation (also inside leave-one-out), and the ratio is reported
        next to an empirical-Bayes shrunk ratio with a 95% interval (shrink_m
        pseudo-docs at the others' composition). Shrinkage changes what is
        displayed, never whether a cell is significant.
    n_needed_1_5x / n_needed_2x: brand docs required to detect a 1.5x / 2x effect
        at the current others' share and n proportion -> weekly power watch.
    Leave-one-out: each significant cell is re-tested with every other brand
        removed from the "others" pool in turn. One brand with an extreme
        composition (e.g. 76% of its docs in one category) shifts the pool and
        makes every other brand look over/under-indexed on that category.
        robust = significant in all leave-one-out tests; only robust cells get
        the 통계 검증 tag. loo_fragile_without lists the brands whose removal
        drops the cell below the threshold; ratio_loo_min/max give the range.
    """
    print("\n" + "=" * 70)
    print(f"Part 2: Brand x Category Proportion Test — segment: {label}")
    print("=" * 70)

    clauses, params = [], []
    if author_types:
        clauses.append("s.author_type = ANY(%s)")
        params.append(author_types)
    if sources:
        clauses.append("s.source = ANY(%s)")
        params.append(sources)
    extra = (" AND " + " AND ".join(clauses)) if clauses else ""
    with get_conn() as conn:
        df = pd.read_sql(
            f"""
                SELECT s.brand, s.topic_label
                FROM staging.content_enriched s
                LEFT JOIN raw.naver_voc v
                  ON s.source IN ('blog', 'cafearticle') AND s.source_id = v.id::text
                WHERE s.topic_label != 'noise' AND s.brand != 'general'
                  AND (v.query IS NULL OR v.query LIKE '%%후기' OR v.query IN ('자동차보험 비교', '실손보험 추천'))
                  {extra}
                """, conn, params=params or None)
    print(f"  (symmetric subset, n={len(df)})")
    print("  (symmetric subset: target-brand product queries excluded from cross-brand test)")

    ct_full = pd.crosstab(df["brand"], df["topic_label"])
    col_totals = ct_full.sum(axis=0)
    sparse_cols = col_totals[col_totals < min_col_n]
    ct = ct_full.drop(columns=sparse_cols.index)
    if len(sparse_cols):
        print(f"  Excluded from test (category n < {min_col_n} in segment): "
              + ", ".join(f"{c}({int(n)})" for c, n in sparse_cols.items()))
    print("  Brand n: " + ", ".join(f"{b}={int(n)}" for b, n in ct.sum(axis=1).items()))

    chi2, p, dof, _ = stats.chi2_contingency(ct)
    print(f"\nOmnibus chi-square: chi2={chi2:.1f}, dof={dof}, p={p:.2e}")

    n_tests = ct.shape[0] * ct.shape[1]
    alpha_bonf = alpha / n_tests
    z_alpha = stats.norm.ppf(1 - alpha_bonf / 2)
    z_power = stats.norm.ppf(power)
    print(f"Post-hoc: {n_tests} cells, Bonferroni alpha = {alpha_bonf:.5f}, MDE power = {power:.0%}")

    rows = []
    for brand in ct.index:
        n_b = int(ct.loc[brand].sum())
        others = ct.drop(index=brand)
        n_o = int(others.values.sum())
        for cat in ct.columns:
            k_b = int(ct.loc[brand, cat])
            k_o = int(others[cat].sum())
            p_b, p_o = k_b / n_b, k_o / n_o

            se_null = np.sqrt(p_o * (1 - p_o) * (1 / n_b + 1 / n_o)) if 0 < p_o < 1 else np.nan
            if np.isfinite(se_null) and se_null > 0:
                delta = (z_alpha + z_power) * se_null
                mde_over = (p_o + delta) / p_o
                mde_under = max(p_o - delta, 0.0) / p_o
            else:
                mde_over, mde_under = np.nan, np.nan

            tested = k_b >= min_cell_n
            small = k_b < small_cell_n
            ratio = z = pval = np.nan
            p_method = ""
            significant, robust, note = False, False, ""
            loo_ratios, loo_fail = [], []
            if tested:
                ratio, z, pval, p_method = _cell_test(k_b, n_b, k_o, n_o, small)
                significant = bool(pval < alpha_bonf)
                if significant:
                    for o in others.index:
                        pool = others.drop(index=o)
                        n_oo = int(pool.values.sum())
                        if n_oo == 0:
                            continue
                        r_o, _, p_oo, _ = _cell_test(k_b, n_b, int(pool[cat].sum()), n_oo, small)
                        loo_ratios.append(r_o)
                        if not (p_oo < alpha_bonf):
                            loo_fail.append(o)
                    robust = len(loo_fail) == 0
            else:
                note = f"n<{min_cell_n}"
            r_shr, ci_lo, ci_hi = _shrunk_ratio(k_b, n_b, p_o, m=shrink_m)
            need_15 = _n_needed(p_o, n_b, n_o, z_alpha + z_power, 1.5)
            need_20 = _n_needed(p_o, n_b, n_o, z_alpha + z_power, 2.0)

            rows.append({
                "brand": brand, "topic_label": cat,
                "n_brand_cat": k_b, "n_brand": n_b, "n_others_cat": k_o, "n_others": n_o,
                "share_brand": round(p_b, 4), "share_others": round(p_o, 4),
                "ratio": round(ratio, 2) if np.isfinite(ratio) else np.nan,
                "z": round(z, 2) if np.isfinite(z) else np.nan,
                "p_value": pval,
                "p_method": p_method,
                "ratio_shrunk": round(r_shr, 2) if np.isfinite(r_shr) else np.nan,
                "ratio_ci_low": round(ci_lo, 2) if np.isfinite(ci_lo) else np.nan,
                "ratio_ci_high": round(ci_hi, 2) if np.isfinite(ci_hi) else np.nan,
                "n_needed_1_5x": need_15,
                "n_needed_2x": need_20,
                "significant": significant,
                "robust": robust,
                "loo_fragile_without": ",".join(loo_fail),
                "ratio_loo_min": round(min(loo_ratios), 2) if loo_ratios else np.nan,
                "ratio_loo_max": round(max(loo_ratios), 2) if loo_ratios else np.nan,
                "mde_ratio_over": round(mde_over, 2) if np.isfinite(mde_over) else np.nan,
                "mde_ratio_under": round(mde_under, 2) if np.isfinite(mde_under) else np.nan,
                "signal_tag": TAG_STAT if robust else TAG_CLUSTER,
                "note": note,
            })
    res = pd.DataFrame(rows)
    res["segment"] = label

    sig = res[res["significant"]].sort_values(["brand", "ratio"], ascending=[True, False])
    print(f"\nSignificant over/under-indexing (Bonferroni, alpha={alpha}): "
          f"{len(sig)} cells, {int(sig['robust'].sum())} robust to leave-one-out")
    for _, r in sig.iterrows():
        direction = "over" if r["ratio"] > 1 else "under"
        mark = " <- TARGET" if r["brand"] == TARGET_BRAND else ""
        loo = ("robust" if r["robust"] else f"fragile: drops without {r['loo_fragile_without']}")
        pm = " (fisher)" if r["p_method"] == "fisher" else ""
        print(f"  {r['brand']:<14} {r['topic_label']:<18} {r['ratio']:>5.2f}x  "
              f"({r['share_brand']:.1%} vs {r['share_others']:.1%})  z={r['z']:+.1f}{pm}  {direction}"
              f"  loo {r['ratio_loo_min']:.2f}-{r['ratio_loo_max']:.2f}x  {loo}{mark}")
    if sig.empty:
        print("  (none)")

    small_cells = res[(res["note"] == "") & (res["n_brand_cat"] < small_cell_n)]
    if len(small_cells):
        print(f"  Small cells (n<{small_cell_n}, Fisher p, shrunk ratio [95% interval]):")
        for _, r in small_cells.sort_values(["brand", "topic_label"]).iterrows():
            print(f"    {r['brand']:<14} {r['topic_label']:<18} n={r['n_brand_cat']:>3}  "
                  f"{r['ratio']:.2f}x -> {r['ratio_shrunk']:.2f}x [{r['ratio_ci_low']:.2f}, {r['ratio_ci_high']:.2f}]")

    small = res[res["significant"] & (res["n_brand_cat"] < 30)]
    if len(small):
        print(f"  Significant cells with n<30 (unstable): "
              + ", ".join(f"{r['brand']}/{r['topic_label']}({r['n_brand_cat']})" for _, r in small.iterrows()))
    under_mde = res[res["significant"] & (
        ((res["ratio"] > 1) & (res["ratio"] < res["mde_ratio_over"]))
        | ((res["ratio"] < 1) & (res["ratio"] > res["mde_ratio_under"])))]
    if len(under_mde):
        print(f"  Significant but below MDE (low power, effect likely overestimated): "
              + ", ".join(f"{r['brand']}/{r['topic_label']}" for _, r in under_mde.iterrows()))
    skipped = res[res["note"] != ""]
    if len(skipped):
        print(f"  Cells not tested (n<{min_cell_n}): "
              + ", ".join(f"{r['brand']}/{r['topic_label']}({r['n_brand_cat']})" for _, r in skipped.iterrows()))

    tgt = res[res["brand"] == TARGET_BRAND].sort_values("topic_label")
    if len(tgt):
        print(f"\nMinimum detectable effect for {TARGET_BRAND} (n={int(tgt['n_brand'].iloc[0])}):")
        for _, r in tgt.iterrows():
            under = f"{r['mde_ratio_under']:.2f}x" if np.isfinite(r["mde_ratio_under"]) and r["mde_ratio_under"] > 0 else "n/a"
            over = f"{r['mde_ratio_over']:.2f}x" if np.isfinite(r["mde_ratio_over"]) else "n/a"
            obs = f"{r['ratio']:.2f}x" if np.isfinite(r["ratio"]) else "n/a"
            print(f"  {r['topic_label']:<18} n={r['n_brand_cat']:>3}  p_others={r['share_others']:.1%}  "
                  f"observed={obs:>6}  detectable: under {under} / over {over}")
        print(f"\nPower watch for {TARGET_BRAND}: brand docs needed in segment (now {int(tgt['n_brand'].iloc[0])})")
        for _, r in tgt.iterrows():
            n15 = f"{int(r['n_needed_1_5x'])}" if np.isfinite(r["n_needed_1_5x"]) else "n/a"
            n20 = f"{int(r['n_needed_2x'])}" if np.isfinite(r["n_needed_2x"]) else "n/a"
            print(f"  {r['topic_label']:<18} for 2x: {n20:>6}   for 1.5x: {n15:>6}")

    res.to_csv(f"{EXPORT_DIR}/brand_category_test_{label}.csv", index=False, encoding="utf-8-sig")
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

    web = ["blog", "cafearticle"]
    r1 = brand_category_test(sources=web, author_types=["customer"], label="web_customer")
    r2 = brand_category_test(sources=web, author_types=["advisor"], label="web_advisor")
    r3 = brand_category_test(sources=["youtube_comment"], label="youtube")
    r0 = brand_category_test(label="all_pooled")  # reference only — source-mix confounded
    pd.concat([r1, r2, r3, r0]).to_csv(f"{EXPORT_DIR}/brand_category_test.csv", index=False, encoding="utf-8-sig")

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
