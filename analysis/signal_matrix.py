"""Analysis: Content Signal Matrix + Lead-Lag pattern exploration.

1. Builds Engagement vs Search Interest matrix per topic category
2. Explores lead-lag patterns using MSTL residual cross-correlation

Usage:
    python -m analysis.signal_matrix
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from config.db import get_conn


# ============================================================
# Part 1: Content Signal Matrix
# ============================================================

def build_signal_matrix():
    """Build Engagement vs Search Interest matrix per brand x topic."""
    print("=" * 70)
    print("Part 1: Content Signal Matrix")
    print("=" * 70)

    # Get content performance with topic labels
    with get_conn() as conn:
        content_df = pd.read_sql(
            """
            SELECT s.brand, s.topic_label, s.published_at,
                   v.view_count, v.like_count, v.comment_count
            FROM staging.content_enriched s
            LEFT JOIN raw.youtube_comments c ON s.source_id = c.comment_id
            LEFT JOIN raw.youtube_videos v ON c.video_id = v.video_id
            WHERE s.topic_label NOT IN ('noise')
              AND s.source = 'youtube_comment'
              AND s.published_at IS NOT NULL
            """,
            conn,
        )

        search_df = pd.read_sql(
            "SELECT brand, week_start, search_ratio, wow_change_pct FROM mart.brand_search_weekly",
            conn,
        )

    content_df["published_at"] = pd.to_datetime(content_df["published_at"])
    search_df["week_start"] = pd.to_datetime(search_df["week_start"])

    # Map brand names for joining
    brand_map = {"삼성생명": "samsung", "한화생명": "hanwha", "교보생명": "kyobo", "메트라이프": "metlife"}
    search_df["brand_key"] = search_df["brand"].map(brand_map)

    # Assign week to content
    content_df["week_start"] = content_df["published_at"].dt.to_period("W-SUN").apply(
        lambda x: x.start_time
    )

    # Engagement per topic category per brand per week
    engagement = content_df.groupby(["brand", "topic_label", "week_start"]).agg(
        total_engagement=("like_count", lambda x: x.fillna(0).sum() + content_df.loc[x.index, "comment_count"].fillna(0).sum()),
        content_count=("brand", "count"),
    ).reset_index()

    # For each brand x topic, calculate average engagement and
    # average search interest change in the week after content posting
    matrix_records = []

    for (brand, topic), group in engagement.groupby(["brand", "topic_label"]):
        avg_engagement = group["total_engagement"].mean()
        sample_count = len(group)

        # Get search interest changes for weeks following this content
        search_signals = []
        for _, row in group.iterrows():
            next_week = row["week_start"] + pd.Timedelta(weeks=1)
            search_match = search_df[
                (search_df["brand_key"] == brand) &
                (search_df["week_start"] == next_week)
            ]
            if not search_match.empty:
                wow = search_match.iloc[0]["wow_change_pct"]
                if pd.notna(wow):
                    search_signals.append(wow)

        avg_search_signal = np.mean(search_signals) if search_signals else 0

        # Determine quadrant
        quadrant = classify_quadrant(avg_engagement, avg_search_signal)

        matrix_records.append({
            "brand": brand,
            "topic_label": topic,
            "avg_engagement": round(avg_engagement, 2),
            "search_signal": round(avg_search_signal, 4),
            "quadrant": quadrant,
            "sample_count": sample_count,
        })

    matrix_df = pd.DataFrame(matrix_records)

    # Print matrix
    print("\nEngagement vs Search Interest Matrix:")
    print(f"{'Brand':<10} {'Topic':<18} {'Engagement':<14} {'Search Signal':<16} {'Quadrant':<12} {'N'}")
    print("-" * 82)
    for _, row in matrix_df.sort_values(["brand", "quadrant"]).iterrows():
        print(f"{row['brand']:<10} {row['topic_label']:<18} "
              f"{row['avg_engagement']:<14.1f} {row['search_signal']:<16.4f} "
              f"{row['quadrant']:<12} {row['sample_count']}")

    return matrix_df


def classify_quadrant(engagement, search_signal):
    """Classify into 4 quadrants based on median thresholds."""
    # Use simple sign-based classification
    # Positive search signal = brand search increased after content
    high_eng = engagement > 100  # threshold adjustable
    high_sig = search_signal > 0

    if high_eng and high_sig:
        return "high-high"
    elif high_eng and not high_sig:
        return "high-low"
    elif not high_eng and high_sig:
        return "low-high"
    else:
        return "low-low"


def save_matrix_to_db(matrix_df):
    """Save matrix to mart.content_signal_matrix."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM mart.content_signal_matrix")
            for _, row in matrix_df.iterrows():
                cur.execute(
                    """
                    INSERT INTO mart.content_signal_matrix
                        (brand, topic_label, avg_engagement, search_signal,
                         quadrant, sample_count)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        row["brand"], row["topic_label"],
                        float(row["avg_engagement"]), float(row["search_signal"]),
                        row["quadrant"], int(row["sample_count"]),
                    ),
                )
    print(f"\nSaved {len(matrix_df)} rows to mart.content_signal_matrix")


# ============================================================
# Part 2: Lead-Lag Analysis (MSTL residual cross-correlation)
# ============================================================

def lead_lag_analysis():
    """Explore lead-lag patterns between content engagement and search interest."""
    print("\n" + "=" * 70)
    print("Part 2: Lead-Lag Analysis (MSTL residual cross-correlation)")
    print("=" * 70)

    from statsmodels.tsa.seasonal import MSTL

    with get_conn() as conn:
        # Weekly content volume per brand
        content_weekly = pd.read_sql(
            """
            SELECT brand, DATE_TRUNC('week', published_at::timestamp)::date AS week_start,
                   COUNT(*) AS content_volume,
                   AVG(CASE WHEN sentiment_label = 'positive' THEN 1
                            WHEN sentiment_label = 'negative' THEN -1
                            ELSE 0 END) AS avg_sentiment
            FROM staging.content_enriched
            WHERE topic_label != 'noise'
              AND published_at IS NOT NULL
            GROUP BY brand, DATE_TRUNC('week', published_at::timestamp)::date
            ORDER BY brand, week_start
            """,
            conn,
        )

        search_weekly = pd.read_sql(
            "SELECT brand, week_start, search_ratio FROM mart.brand_search_weekly ORDER BY brand, week_start",
            conn,
        )

    content_weekly["week_start"] = pd.to_datetime(content_weekly["week_start"])
    search_weekly["week_start"] = pd.to_datetime(search_weekly["week_start"])

    brand_map = {"삼성생명": "samsung", "한화생명": "hanwha", "교보생명": "kyobo", "메트라이프": "metlife"}
    search_weekly["brand_key"] = search_weekly["brand"].map(brand_map)

    results = {}

    for brand_key in ["samsung", "hanwha", "kyobo", "metlife"]:
        cdf = content_weekly[content_weekly["brand"] == brand_key].set_index("week_start").sort_index()
        sdf = search_weekly[search_weekly["brand_key"] == brand_key].set_index("week_start").sort_index()

        if len(cdf) < 10 or len(sdf) < 10:
            print(f"\n  {brand_key}: insufficient data (content={len(cdf)}, search={len(sdf)})")
            continue

        # Align on common weeks
        common_idx = cdf.index.intersection(sdf.index)
        if len(common_idx) < 10:
            print(f"\n  {brand_key}: insufficient overlapping weeks ({len(common_idx)})")
            continue

        content_series = cdf.loc[common_idx, "content_volume"].astype(float)
        search_series = sdf.loc[common_idx, "search_ratio"].astype(float)

        # MSTL decomposition to remove seasonality
        try:
            # Search series decomposition
            if len(search_series) >= 14:
                mstl_search = MSTL(search_series, periods=[4, 13]).fit()
                search_resid = mstl_search.resid
            else:
                search_resid = search_series - search_series.mean()

            # Content series: simple detrend (usually too short for MSTL)
            content_resid = content_series - content_series.rolling(4, min_periods=1).mean()

        except Exception as e:
            print(f"\n  {brand_key}: MSTL error: {e}")
            search_resid = search_series - search_series.mean()
            content_resid = content_series - content_series.mean()

        # Cross-correlation at different lags
        max_lag = min(8, len(common_idx) // 3)
        lag_corrs = {}

        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                corr = content_resid.corr(search_resid)
            elif lag > 0:
                # Positive lag: content leads search by 'lag' weeks
                corr = content_resid.iloc[:-lag].reset_index(drop=True).corr(
                    search_resid.iloc[lag:].reset_index(drop=True)
                )
            else:
                # Negative lag: search leads content
                corr = content_resid.iloc[-lag:].reset_index(drop=True).corr(
                    search_resid.iloc[:lag].reset_index(drop=True)
                )

            if pd.notna(corr):
                lag_corrs[lag] = round(corr, 4)

        results[brand_key] = lag_corrs

        # Print
        print(f"\n  {brand_key} (n={len(common_idx)} weeks):")
        print(f"    Lag   Correlation   Interpretation")
        print(f"    {'---':<6}{'---':<14}{'---'}")
        for lag, corr in sorted(lag_corrs.items()):
            direction = ""
            if lag > 0:
                direction = f"content leads search by {lag}w"
            elif lag < 0:
                direction = f"search leads content by {abs(lag)}w"
            else:
                direction = "contemporaneous"

            marker = " *" if abs(corr) > 0.3 else ""
            print(f"    {lag:<6}{corr:<14.4f}{direction}{marker}")

        # Best lag
        if lag_corrs:
            best_lag = max(lag_corrs, key=lambda k: abs(lag_corrs[k]))
            print(f"    -> Strongest signal: lag={best_lag}, r={lag_corrs[best_lag]:.4f}")

    return results


# ============================================================
# Main
# ============================================================

def run():
    # Part 1
    matrix_df = build_signal_matrix()
    save_matrix_to_db(matrix_df)

    # Save CSV
    os.makedirs("data/exports", exist_ok=True)
    matrix_df.to_csv("data/exports/content_signal_matrix.csv", index=False, encoding="utf-8-sig")
    print("CSV exported: data/exports/content_signal_matrix.csv")

    # Part 2
    lag_results = lead_lag_analysis()

    # Save lag results
    import json
    with open("data/exports/lead_lag_results.json", "w", encoding="utf-8") as f:
        json.dump(lag_results, f, ensure_ascii=False, indent=2)
    print("\nLead-lag results saved: data/exports/lead_lag_results.json")

    print("\n" + "=" * 70)
    print("Phase 4 complete.")
    print("  Outputs:")
    print("    - mart.content_signal_matrix (DB)")
    print("    - data/exports/content_signal_matrix.csv")
    print("    - data/exports/lead_lag_results.json")


if __name__ == "__main__":
    run()
