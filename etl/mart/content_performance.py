"""ETL: raw.youtube_videos -> mart.content_performance

Computes weekly engagement metrics per brand.
Subscriber counts fetched live from YouTube API (fallback: pre-EDA JSON).
Topic labels are NULL at this stage -- populated after the NLP stage.

Usage:
    python -m etl.mart.content_performance
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
import json
import pandas as pd
from dotenv import load_dotenv
from config.db import get_conn
from config.settings import BRANDS

load_dotenv()


def get_subscriber_counts():
    """Fetch subscriber counts per brand (name_kr) from YouTube API."""
    counts = {}
    try:
        from googleapiclient.discovery import build
        yt = build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))
        for key, info in BRANDS.items():
            cid = info.get("youtube_channel_id")
            if not cid:
                continue
            resp = yt.channels().list(part="statistics", id=cid).execute()
            if resp.get("items"):
                counts[info["name_kr"]] = int(resp["items"][0]["statistics"].get("subscriberCount", 0))
        print(f"Subscriber counts (live): {counts}")
        return counts
    except Exception as e:
        print(f"[WARN] YouTube API unavailable ({e}); falling back to pre-EDA JSON")

    path = "data/raw/pre_eda_youtube_channels.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key, cands in data.items():
            cid = BRANDS[key].get("youtube_channel_id")
            for c in cands:
                if c["channel_id"] == cid:
                    counts[BRANDS[key]["name_kr"]] = c["subscribers"]
    return counts


def extract():
    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT brand, video_id, title, published_at, view_count, like_count, comment_count
            FROM raw.youtube_videos ORDER BY brand, published_at
            """, conn)
    df["published_at"] = pd.to_datetime(df["published_at"])
    print(f"Extracted {len(df)} videos from raw.youtube_videos")
    return df


def transform(df):
    subs = get_subscriber_counts()
    df["week_start"] = df["published_at"].dt.to_period("W-SUN").apply(lambda x: x.start_time)
    df["engagement_rate"] = df.apply(
        lambda r: (r["like_count"] + r["comment_count"]) / max(subs.get(r["brand"], 1), 1) * 100, axis=1)

    weekly = df.groupby(["brand", "week_start"]).agg(
        content_count=("video_id", "count"),
        avg_engagement=("engagement_rate", "mean"),
        total_views=("view_count", "sum"),
        total_likes=("like_count", "sum"),
        total_comments=("comment_count", "sum"),
    ).reset_index()
    weekly["avg_engagement"] = weekly["avg_engagement"].round(4)
    weekly["source"] = "youtube"
    weekly["topic_id"] = None
    weekly["topic_label"] = None
    print(f"Transformed to {len(weekly)} weekly records")
    return weekly


def load(df):
    with get_conn() as conn:
        with conn.cursor() as cur:
            for _, row in df.iterrows():
                cur.execute(
                    """
                    INSERT INTO mart.content_performance
                        (brand, week_start, source, topic_id, topic_label,
                         content_count, avg_engagement, total_views, total_likes, total_comments)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (brand, week_start, source, topic_id)
                    DO UPDATE SET content_count = EXCLUDED.content_count,
                                  avg_engagement = EXCLUDED.avg_engagement,
                                  total_views = EXCLUDED.total_views,
                                  total_likes = EXCLUDED.total_likes,
                                  total_comments = EXCLUDED.total_comments
                    """,
                    (row["brand"], row["week_start"].date(), row["source"], row["topic_id"],
                     row["topic_label"], int(row["content_count"]), float(row["avg_engagement"]),
                     int(row["total_views"]), int(row["total_likes"]), int(row["total_comments"])),
                )
    print(f"Loaded {len(df)} rows to mart.content_performance")


def save_csv(df, path="data/exports/content_performance.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"CSV exported: {path}")


def run():
    result = transform(extract())
    load(result)
    save_csv(result)
    print("\nWeekly content summary by brand:")
    for brand in sorted(result["brand"].unique()):
        bdf = result[result["brand"] == brand]
        print(f"  {brand}: {len(bdf)} weeks, avg {bdf['content_count'].mean():.1f} videos/week, "
              f"avg engagement {bdf['avg_engagement'].mean():.4f}%")


if __name__ == "__main__":
    run()
