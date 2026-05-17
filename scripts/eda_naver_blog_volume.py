"""Pre-EDA Check 1: Naver Blog/Cafe VoC volume per brand.

Checklist item: "메트라이프 보험" 네이버 블로그 볼륨 확인
  -> 월 50건 미만이면 Cafe/YouTube 댓글로 전환

Usage:
    python scripts/eda_naver_blog_volume.py
"""

import os
import json
import time
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

HEADERS = {
    "X-Naver-Client-Id": CLIENT_ID,
    "X-Naver-Client-Secret": CLIENT_SECRET,
}

SEARCH_QUERIES = {
    "metlife_blog": [
        "메트라이프 보험 후기",
        "메트라이프생명",
        "메트라이프 보험",
    ],
    "samsung_blog": [
        "삼성생명 후기",
        "삼성생명 보험",
    ],
    "hanwha_blog": [
        "한화생명 후기",
        "한화생명 보험",
    ],
    "kyobo_blog": [
        "교보생명 후기",
        "교보생명 보험",
    ],
}

# Naver Search API sources
SOURCES = ["blog", "cafearticle"]


def search_naver(query, source, display=10, start=1):
    """Call Naver Search API and return total count."""
    url = f"https://openapi.naver.com/v1/search/{source}.json"
    params = {
        "query": query,
        "display": display,
        "start": start,
        "sort": "date",
    }
    resp = requests.get(url, headers=HEADERS, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("total", 0), data.get("items", [])


def estimate_monthly_volume(query, source):
    """Estimate monthly volume by checking recent 30-day posts.

    Naver Search API does not support date filtering directly,
    so we fetch recent results and check posting dates.
    """
    total, items = search_naver(query, source, display=100, start=1)

    if not items:
        return 0, total

    # Count items within last 30 days
    cutoff = datetime.now() - timedelta(days=30)
    recent_count = 0
    for item in items:
        post_date_str = item.get("postdate", "")
        if post_date_str:
            try:
                post_date = datetime.strptime(post_date_str, "%Y%m%d")
                if post_date >= cutoff:
                    recent_count += 1
            except ValueError:
                pass

    return recent_count, total


def run_volume_check():
    """Run volume check for all brands and sources."""
    if not CLIENT_ID or not CLIENT_SECRET:
        print("[ERROR] NAVER_CLIENT_ID / NAVER_CLIENT_SECRET not set in .env")
        print("  Get API keys: https://developers.naver.com/apps/#/register")
        return

    results = {}

    for brand, queries in SEARCH_QUERIES.items():
        brand_name = brand.split("_")[0]
        results[brand_name] = {}

        for source in SOURCES:
            source_total = 0
            source_monthly = 0

            for query in queries:
                monthly, total = estimate_monthly_volume(query, source)
                source_monthly += monthly
                source_total += total
                time.sleep(0.1)  # rate limit

            results[brand_name][source] = {
                "monthly_estimate": source_monthly,
                "total_available": source_total,
            }

    # Print results
    print("=" * 70)
    print("Pre-EDA Check 1: Naver Blog/Cafe VoC Volume")
    print("=" * 70)
    print(f"{'Brand':<12} {'Source':<15} {'Monthly Est.':<15} {'Total':<10} {'Status'}")
    print("-" * 70)

    for brand, sources in results.items():
        for source, counts in sources.items():
            monthly = counts["monthly_estimate"]
            total = counts["total_available"]
            status = "OK" if monthly >= 50 else "LOW -- consider fallback"
            print(f"{brand:<12} {source:<15} {monthly:<15} {total:<10} {status}")
        print()

    # Decision summary
    print("=" * 70)
    print("Decision Points:")
    metlife_blog = results.get("metlife", {}).get("blog", {}).get("monthly_estimate", 0)
    metlife_cafe = results.get("metlife", {}).get("cafearticle", {}).get("monthly_estimate", 0)
    metlife_total = metlife_blog + metlife_cafe

    if metlife_total < 50:
        print(f"  MetLife Blog+Cafe monthly: {metlife_total} (< 50)")
        print("  -> FALLBACK: YouTube comments + Naver Cafe as primary VoC source")
    else:
        print(f"  MetLife Blog+Cafe monthly: {metlife_total} (>= 50)")
        print("  -> PROCEED: Blog + Cafe as VoC source")

    # Save raw results
    out_path = "data/raw/pre_eda_blog_volume.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Raw results saved: {out_path}")


if __name__ == "__main__":
    run_volume_check()
