"""Collector: YouTube official channel videos + comments.

Fetches videos and comments from 3 insurance brand channels
(MetLife excluded -- no official channel).
Stores in raw.youtube_videos and raw.youtube_comments.

Usage:
    python -m collectors.youtube
    python -m collectors.youtube --brand samsung
    python -m collectors.youtube --max-videos 100
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import argparse
import time
import json

import pandas as pd
from dotenv import load_dotenv

from config.settings import BRANDS, YOUTUBE
from config.db import get_conn, _use_csv_fallback

load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")


def get_youtube_service():
    """Build YouTube API client."""
    from googleapiclient.discovery import build

    if not API_KEY:
        print("[ERROR] YOUTUBE_API_KEY not set")
        return None
    return build("youtube", "v3", developerKey=API_KEY)


def fetch_videos(youtube, channel_id, max_results=50):
    """Fetch recent videos from a channel."""
    videos = []
    next_page = None

    while len(videos) < max_results:
        request = youtube.search().list(
            part="snippet",
            channelId=channel_id,
            type="video",
            order="date",
            maxResults=min(50, max_results - len(videos)),
            pageToken=next_page,
        )
        response = request.execute()

        video_ids = [item["id"]["videoId"] for item in response.get("items", [])]
        if not video_ids:
            break

        # Get video statistics
        stats_resp = youtube.videos().list(
            part="statistics,snippet,contentDetails",
            id=",".join(video_ids),
        ).execute()

        for item in stats_resp.get("items", []):
            stats = item.get("statistics", {})
            videos.append({
                "video_id": item["id"],
                "channel_id": channel_id,
                "title": item["snippet"]["title"],
                "description": item["snippet"].get("description", ""),
                "published_at": item["snippet"]["publishedAt"],
                "view_count": int(stats.get("viewCount", 0)),
                "like_count": int(stats.get("likeCount", 0)),
                "comment_count": int(stats.get("commentCount", 0)),
            })

        next_page = response.get("nextPageToken")
        if not next_page:
            break
        time.sleep(0.2)

    return videos


def fetch_comments(youtube, video_id, max_results=100):
    """Fetch top-level comments for a video."""
    comments = []
    next_page = None

    try:
        while len(comments) < max_results:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                order="relevance",
                maxResults=min(100, max_results - len(comments)),
                pageToken=next_page,
            )
            response = request.execute()

            for item in response.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "comment_id": item["id"],
                    "video_id": video_id,
                    "author": snippet.get("authorDisplayName", ""),
                    "text_original": snippet.get("textOriginal", ""),
                    "like_count": snippet.get("likeCount", 0),
                    "published_at": snippet.get("publishedAt", ""),
                })

            next_page = response.get("nextPageToken")
            if not next_page:
                break
            time.sleep(0.1)

    except Exception as e:
        if "commentsDisabled" in str(e):
            pass  # skip videos with comments disabled
        else:
            print(f"  [WARN] Comments error for {video_id}: {e}")

    return comments


def save_videos_to_db(videos, brand):
    """Insert videos into raw.youtube_videos."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            for v in videos:
                cur.execute(
                    """
                    INSERT INTO raw.youtube_videos
                        (video_id, channel_id, brand, title, description,
                         published_at, view_count, like_count, comment_count)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (video_id)
                    DO UPDATE SET view_count = EXCLUDED.view_count,
                                  like_count = EXCLUDED.like_count,
                                  comment_count = EXCLUDED.comment_count,
                                  collected_at = NOW()
                    """,
                    (
                        v["video_id"], v["channel_id"], brand,
                        v["title"], v["description"], v["published_at"],
                        v["view_count"], v["like_count"], v["comment_count"],
                    ),
                )
    print(f"  DB: {len(videos)} videos upserted for {brand}")


def save_comments_to_db(comments):
    """Insert comments into raw.youtube_comments."""
    if not comments:
        return
    with get_conn() as conn:
        with conn.cursor() as cur:
            for c in comments:
                cur.execute(
                    """
                    INSERT INTO raw.youtube_comments
                        (comment_id, video_id, author, text_original,
                         like_count, published_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (comment_id) DO NOTHING
                    """,
                    (
                        c["comment_id"], c["video_id"], c["author"],
                        c["text_original"], c["like_count"], c["published_at"],
                    ),
                )
    print(f"  DB: {len(comments)} comments inserted")


def save_to_csv(videos, comments, brand):
    """Save to CSV fallback."""
    os.makedirs("data/raw", exist_ok=True)

    if videos:
        vdf = pd.DataFrame(videos)
        vdf["brand"] = brand
        vpath = "data/raw/youtube_videos.csv"
        if os.path.exists(vpath):
            existing = pd.read_csv(vpath)
            vdf = pd.concat([existing, vdf]).drop_duplicates(
                subset=["video_id"], keep="last"
            )
        vdf.to_csv(vpath, index=False, encoding="utf-8-sig")
        print(f"  CSV: {len(vdf)} videos saved to {vpath}")

    if comments:
        cdf = pd.DataFrame(comments)
        cpath = "data/raw/youtube_comments.csv"
        if os.path.exists(cpath):
            existing = pd.read_csv(cpath)
            cdf = pd.concat([existing, cdf]).drop_duplicates(
                subset=["comment_id"], keep="last"
            )
        cdf.to_csv(cpath, index=False, encoding="utf-8-sig")
        print(f"  CSV: {len(cdf)} comments saved to {cpath}")


def collect(brand_filter=None, max_videos=None):
    """Main collection entry point."""
    youtube = get_youtube_service()
    if not youtube:
        return

    max_vids = max_videos or YOUTUBE["max_results_per_channel"]
    max_comments = YOUTUBE["max_comments_per_video"]

    for brand_key, brand_info in BRANDS.items():
        if not brand_info.get("youtube_available", False):
            print(f"Skipping {brand_info['name_en']} (no YouTube channel)")
            continue

        if brand_filter and brand_key != brand_filter:
            continue

        channel_id = brand_info["youtube_channel_id"]
        brand_name = brand_info["name_kr"]
        print(f"\nCollecting: {brand_info['name_en']} ({channel_id})")

        # Fetch videos
        videos = fetch_videos(youtube, channel_id, max_results=max_vids)
        print(f"  Fetched {len(videos)} videos")

        # Fetch comments for each video
        all_comments = []
        for i, video in enumerate(videos):
            if video["comment_count"] > 0:
                comments = fetch_comments(
                    youtube, video["video_id"], max_results=max_comments
                )
                all_comments.extend(comments)
                if (i + 1) % 10 == 0:
                    print(f"  Comments progress: {i + 1}/{len(videos)} videos")
            time.sleep(0.1)

        print(f"  Total comments fetched: {len(all_comments)}")

        # Save
        if _use_csv_fallback():
            save_to_csv(videos, all_comments, brand_name)
        else:
            save_videos_to_db(videos, brand_name)
            save_comments_to_db(all_comments)
            save_to_csv(videos, all_comments, brand_name)


def main():
    parser = argparse.ArgumentParser(description="Collect YouTube videos + comments")
    parser.add_argument("--brand", default=None, help="Single brand key to collect")
    parser.add_argument("--max-videos", type=int, default=None, help="Max videos per channel")
    args = parser.parse_args()

    collect(brand_filter=args.brand, max_videos=args.max_videos)


if __name__ == "__main__":
    main()
