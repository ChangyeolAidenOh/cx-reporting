"""Pre-EDA Check 2: YouTube official channel stats for 4 brands.

Checklist item: 4사 YouTube 공식 채널 영상 수 + 평균 댓글 수 확인

Usage:
    python scripts/eda_youtube_channels.py
"""

import os
import json
import time

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")

# Known official YouTube channel handles/IDs
# Fill these in after manual lookup or use search
CHANNEL_CANDIDATES = {
    "MetLife Korea": {
        "search_query": "메트라이프생명 공식",
        "channel_id": "",  # fill after discovery
    },
    "Samsung Life": {
        "search_query": "삼성생명 공식",
        "channel_id": "",
    },
    "Hanwha Life": {
        "search_query": "한화생명 공식",
        "channel_id": "",
    },
    "Kyobo Life": {
        "search_query": "교보생명 공식",
        "channel_id": "",
    },
}


def get_youtube_service():
    """Build YouTube API client."""
    try:
        from googleapiclient.discovery import build
    except ImportError:
        print("[ERROR] google-api-python-client not installed.")
        print("  pip install google-api-python-client")
        return None

    if not API_KEY:
        print("[ERROR] YOUTUBE_API_KEY not set in .env")
        print("  Get API key: https://console.cloud.google.com/apis/credentials")
        return None

    return build("youtube", "v3", developerKey=API_KEY)


def find_channel(youtube, search_query):
    """Search for an official channel by query."""
    request = youtube.search().list(
        part="snippet",
        q=search_query,
        type="channel",
        maxResults=3,
    )
    response = request.execute()

    channels = []
    for item in response.get("items", []):
        channels.append({
            "channel_id": item["snippet"]["channelId"],
            "title": item["snippet"]["title"],
            "description": item["snippet"]["description"][:100],
        })
    return channels


def get_channel_stats(youtube, channel_id):
    """Get channel statistics."""
    request = youtube.channels().list(
        part="statistics,snippet,contentDetails",
        id=channel_id,
    )
    response = request.execute()

    if not response.get("items"):
        return None

    item = response["items"][0]
    stats = item.get("statistics", {})
    return {
        "title": item["snippet"]["title"],
        "subscriber_count": int(stats.get("subscriberCount", 0)),
        "video_count": int(stats.get("videoCount", 0)),
        "view_count": int(stats.get("viewCount", 0)),
    }


def get_recent_videos(youtube, channel_id, max_results=20):
    """Get recent videos and their comment counts."""
    request = youtube.search().list(
        part="snippet",
        channelId=channel_id,
        type="video",
        order="date",
        maxResults=max_results,
    )
    response = request.execute()

    video_ids = [item["id"]["videoId"] for item in response.get("items", [])]

    if not video_ids:
        return []

    # Get video statistics
    stats_request = youtube.videos().list(
        part="statistics,snippet",
        id=",".join(video_ids),
    )
    stats_response = stats_request.execute()

    videos = []
    for item in stats_response.get("items", []):
        stats = item.get("statistics", {})
        videos.append({
            "title": item["snippet"]["title"][:60],
            "published_at": item["snippet"]["publishedAt"],
            "view_count": int(stats.get("viewCount", 0)),
            "like_count": int(stats.get("likeCount", 0)),
            "comment_count": int(stats.get("commentCount", 0)),
        })

    return videos


def run_youtube_check():
    """Run YouTube channel check for all brands."""
    youtube = get_youtube_service()
    if not youtube:
        return

    results = {}

    for brand, config in CHANNEL_CANDIDATES.items():
        print(f"\nChecking: {brand}")

        # Step 1: Find channel
        channel_id = config.get("channel_id")
        if not channel_id:
            candidates = find_channel(youtube, config["search_query"])
            if candidates:
                print(f"  Found {len(candidates)} channel candidates:")
                for i, c in enumerate(candidates):
                    print(f"    [{i}] {c['title']} ({c['channel_id']})")
                channel_id = candidates[0]["channel_id"]
                print(f"  Using: {candidates[0]['title']}")
            else:
                print(f"  No channels found for query: {config['search_query']}")
                continue

        time.sleep(0.2)

        # Step 2: Get channel stats
        stats = get_channel_stats(youtube, channel_id)
        if not stats:
            print(f"  Could not fetch stats for {channel_id}")
            continue

        time.sleep(0.2)

        # Step 3: Get recent videos
        videos = get_recent_videos(youtube, channel_id, max_results=20)

        avg_comments = 0
        avg_likes = 0
        avg_views = 0
        if videos:
            avg_comments = sum(v["comment_count"] for v in videos) / len(videos)
            avg_likes = sum(v["like_count"] for v in videos) / len(videos)
            avg_views = sum(v["view_count"] for v in videos) / len(videos)

        results[brand] = {
            "channel_id": channel_id,
            "channel_title": stats["title"],
            "subscribers": stats["subscriber_count"],
            "total_videos": stats["video_count"],
            "total_views": stats["view_count"],
            "recent_20_avg_comments": round(avg_comments, 1),
            "recent_20_avg_likes": round(avg_likes, 1),
            "recent_20_avg_views": round(avg_views, 1),
            "recent_videos_sample": videos[:3],
        }

        time.sleep(0.5)

    # Print summary
    print("\n" + "=" * 80)
    print("Pre-EDA Check 2: YouTube Channel Stats")
    print("=" * 80)
    print(f"{'Brand':<18} {'Subscribers':<14} {'Videos':<10} {'Avg Comments':<15} {'Avg Views'}")
    print("-" * 80)

    for brand, data in results.items():
        print(
            f"{brand:<18} "
            f"{data['subscribers']:<14,} "
            f"{data['total_videos']:<10} "
            f"{data['recent_20_avg_comments']:<15} "
            f"{data['recent_20_avg_views']:,.0f}"
        )

    # Decision
    print("\n" + "=" * 80)
    print("Decision Points:")
    for brand, data in results.items():
        avg_c = data["recent_20_avg_comments"]
        if avg_c < 5:
            print(f"  {brand}: avg comments {avg_c} (LOW) -- limited VoC from comments")
        else:
            print(f"  {brand}: avg comments {avg_c} -- usable for VoC")

    # Save
    out_path = "data/raw/pre_eda_youtube_channels.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Raw results saved: {out_path}")


if __name__ == "__main__":
    run_youtube_check()
