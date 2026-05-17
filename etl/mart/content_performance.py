"""ETL: raw.youtube_videos -> mart.content_performance

Computes weekly engagement metrics per brand.
Topic labels are NULL at this stage -- populated after Phase 3 NLP.

Usage:
    python -m etl.mart.content_performance
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from config.db import get_conn
from config.settings import BRANDS


def get_subscriber_counts():
    """Get subscriber counts for engagement rate calculation."""
    # From pre-EDA results (2026-05-14)
    return {
        "삼성생명": 299000,
        "한화생명": 327000,
        "교보생명": 121000,
    }


def extract():
    """Load raw YouTube video data."""
    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT brand, video_id, title, published_at,
                   view_count, like_count, comment_count
            FROM raw.youtube_videos
            ORDER BY brand, published_at
            """,
            conn,
        )
    df["published_at"] = pd.to_datetime(df["published_at"])
    print(f"Extracted {len(df)} videos from raw.youtube_videos")
    return df


def transform(df):
    """Compute weekly content performance metrics."""
    subs = get_subscriber_counts()

    # Assign week_start (Monday)
    df["week_start"] = df["published_at"].dt.to_period("W-SUN").apply(lambda x: x.start_time)

    # Engagement rate per video
    df["engagement_rate"] = df.apply(
        lambda row: (row["like_count"] + row["comment_count"])
        / subs.get(row["brand"], 1) * 100,
        axis=1,
    )

    # Weekly aggregation per brand
    weekly = df.groupby(["brand", "week_start"]).agg(
        content_count=("video_id", "count"),
        avg_engagement=("engagement_rate", "mean"),
        total_views=("view_count", "sum"),
        total_likes=("like_count", "sum"),
        total_comments=("comment_count", "sum"),
    ).reset_index()

    weekly["avg_engagement"] = weekly["avg_engagement"].round(4)

    # Source is YouTube, topic_id/label NULL for now
    weekly["source"] = "youtube"
    weekly["topic_id"] = None
    weekly["topic_label"] = None

    print(f"Transformed to {len(weekly)} weekly records")
    return weekly


def load(df):
    """Upsert into mart.content_performance."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            for _, row in df.iterrows():
                cur.execute(
                    """
                    INSERT INTO mart.content_performance
                        (brand, week_start, source, topic_id, topic_label,
                         content_count, avg_engagement, total_views,
                         total_likes, total_comments)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (brand, week_start, source, topic_id)
                    DO UPDATE SET content_count = EXCLUDED.content_count,
                                  avg_engagement = EXCLUDED.avg_engagement,
                                  total_views = EXCLUDED.total_views,
                                  total_likes = EXCLUDED.total_likes,
                                  total_comments = EXCLUDED.total_comments
                    """,
                    (
                        row["brand"],
                        row["week_start"].date(),
                        row["source"],
                        row["topic_id"],
                        row["topic_label"],
                        int(row["content_count"]),
                        float(row["avg_engagement"]),
                        int(row["total_views"]),
                        int(row["total_likes"]),
                        int(row["total_comments"]),
                    ),
                )
    print(f"Loaded {len(df)} rows to mart.content_performance")


def save_csv(df, path="data/exports/content_performance.csv"):
    """CSV export."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"CSV exported: {path}")


def run():
    df = extract()
    result = transform(df)
    load(result)
    save_csv(result)

    # Summary
    print("\nWeekly content summary by brand:")
    for brand in sorted(result["brand"].unique()):
        bdf = result[result["brand"] == brand]
        print(f"  {brand}: {len(bdf)} weeks, "
              f"avg {bdf['content_count'].mean():.1f} videos/week, "
              f"avg engagement {bdf['avg_engagement'].mean():.4f}%")


if __name__ == "__main__":
    run()
