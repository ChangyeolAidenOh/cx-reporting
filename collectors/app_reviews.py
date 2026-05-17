"""Collector: App store reviews (Google Play + App Store).

Collects MetLife ONE and competitor app reviews.
Google Play: google-play-scraper library
App Store: iTunes RSS API (no extra dependency)

Usage:
    python -m collectors.app_reviews
    python -m collectors.app_reviews --discover
    python -m collectors.app_reviews --store google_play --count 200
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
import pandas as pd
from dotenv import load_dotenv
from config.db import get_conn, _use_csv_fallback

load_dotenv()

# App IDs - fill after discovery
APP_IDS = {
    "metlife": {
        "google_play": "kr.co.metlife.metlifeone",  # 검색에서 None으로 나왔지만 이 ID 시도
        "app_store": "1234737021",  # MetLife One, rating 2.51, 74 reviews
        "name": "MetLife ONE",
    },
    "metlife_360health": {
        "google_play": "com.thecarevoice.metlife",  # 360Health
        "app_store": "",
        "name": "MetLife 360Health",
    },
}

# ============================================================
# App Discovery
# ============================================================

def discover_apps():
    """Search for insurance apps on Google Play."""
    from google_play_scraper import search

    queries = ["메트라이프", "MetLife ONE", "메트라이프 보험"]
    print("Discovering apps on Google Play...\n")

    seen = set()
    for query in queries:
        print(f"  Query: '{query}'")
        try:
            results = search(query, lang="ko", country="kr", n_hits=5)
            for app in results:
                if app["appId"] not in seen:
                    seen.add(app["appId"])
                    print(f"    {app['appId']}")
                    print(f"      Title: {app['title']}")
                    print(f"      Developer: {app.get('developer', 'N/A')}")
                    print(f"      Score: {app.get('score', 'N/A')}")
                    print(f"      Installs: {app.get('installs', 'N/A')}")
                    print()
        except Exception as e:
            print(f"    [WARN] Search error: {e}")
        time.sleep(0.5)

    # App Store discovery via iTunes Search API
    print("\nDiscovering apps on App Store...")
    for query in ["메트라이프", "MetLife"]:
        print(f"  Query: '{query}'")
        try:
            resp = requests.get(
                "https://itunes.apple.com/search",
                params={"term": query, "country": "kr", "entity": "software", "limit": 5},
                timeout=10,
            )
            data = resp.json()
            for app in data.get("results", []):
                print(f"    {app['bundleId']}")
                print(f"      Title: {app['trackName']}")
                print(f"      App Store ID: {app['trackId']}")
                print(f"      Rating: {app.get('averageUserRating', 'N/A')}")
                print(f"      Reviews: {app.get('userRatingCount', 'N/A')}")
                print()
        except Exception as e:
            print(f"    [WARN] Search error: {e}")
        time.sleep(0.5)


# ============================================================
# Google Play Reviews
# ============================================================

def fetch_google_play_reviews(app_id, count=200):
    """Fetch reviews from Google Play."""
    from google_play_scraper import reviews, Sort

    print(f"  Fetching Google Play reviews for {app_id}...")

    all_reviews = []
    continuation_token = None

    while len(all_reviews) < count:
        batch_size = min(100, count - len(all_reviews))
        try:
            result, continuation_token = reviews(
                app_id,
                lang="ko",
                country="kr",
                sort=Sort.NEWEST,
                count=batch_size,
                continuation_token=continuation_token,
            )
            if not result:
                break

            for r in result:
                all_reviews.append({
                    "app_id": app_id,
                    "store": "google_play",
                    "review_id": r.get("reviewId", ""),
                    "author": r.get("userName", ""),
                    "rating": r.get("score", 0),
                    "text_original": r.get("content", ""),
                    "posted_at": r.get("at"),
                })

            if not continuation_token:
                break
            time.sleep(0.3)

        except Exception as e:
            print(f"    [WARN] Error: {e}")
            break

    print(f"    Fetched {len(all_reviews)} reviews")
    return all_reviews


# ============================================================
# App Store Reviews (iTunes RSS)
# ============================================================

def fetch_app_store_reviews(app_id, pages=10):
    """Fetch reviews from App Store using iTunes RSS API."""
    print(f"  Fetching App Store reviews for {app_id}...")

    all_reviews = []

    for page in range(1, pages + 1):
        url = f"https://itunes.apple.com/kr/rss/customerreviews/id={app_id}/page={page}/sortby=mostrecent/json"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                break

            data = resp.json()
            entries = data.get("feed", {}).get("entry", [])

            if not entries:
                break

            # First entry is app metadata, skip it
            for entry in entries:
                if "im:rating" not in entry:
                    continue

                review_id = entry.get("id", {}).get("label", "")
                all_reviews.append({
                    "app_id": str(app_id),
                    "store": "app_store",
                    "review_id": review_id,
                    "author": entry.get("author", {}).get("name", {}).get("label", ""),
                    "rating": int(entry.get("im:rating", {}).get("label", 0)),
                    "text_original": entry.get("content", {}).get("label", ""),
                    "posted_at": None,  # RSS doesn't always include date
                })

            time.sleep(0.5)

        except Exception as e:
            print(f"    [WARN] Page {page} error: {e}")
            break

    print(f"    Fetched {len(all_reviews)} reviews")
    return all_reviews


# ============================================================
# Save
# ============================================================

def save_to_db(reviews, brand):
    """Insert into raw.app_reviews."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            inserted = 0
            for r in reviews:
                try:
                    cur.execute(
                        """
                        INSERT INTO raw.app_reviews
                            (app_id, store, review_id, brand, author,
                             rating, text_original, posted_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (store, review_id) DO NOTHING
                        """,
                        (
                            r["app_id"], r["store"], r["review_id"], brand,
                            r["author"], r["rating"], r["text_original"],
                            r["posted_at"],
                        ),
                    )
                    inserted += 1
                except Exception:
                    pass
    print(f"  DB: {inserted} reviews inserted for {brand}")


def save_to_csv(reviews, brand, path="data/raw/app_reviews.csv"):
    """Save to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(reviews)
    df["brand"] = brand

    if os.path.exists(path):
        existing = pd.read_csv(path)
        df = pd.concat([existing, df]).drop_duplicates(
            subset=["store", "review_id"], keep="last"
        )

    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  CSV: {len(df)} reviews saved to {path}")


# ============================================================
# Main
# ============================================================

def collect(store_filter=None, count=200):
    """Collect app reviews for all configured apps."""
    for brand, config in APP_IDS.items():
        print(f"\nCollecting: {config['name']} ({brand})")

        all_reviews = []

        # Google Play
        if (not store_filter or store_filter == "google_play") and config.get("google_play"):
            gp_reviews = fetch_google_play_reviews(config["google_play"], count=count)
            all_reviews.extend(gp_reviews)

        # App Store
        if (not store_filter or store_filter == "app_store") and config.get("app_store"):
            as_reviews = fetch_app_store_reviews(config["app_store"])
            all_reviews.extend(as_reviews)

        if not all_reviews:
            print(f"  No reviews collected for {brand}")
            continue

        # Save
        if _use_csv_fallback():
            save_to_csv(all_reviews, brand)
        else:
            save_to_db(all_reviews, brand)
            save_to_csv(all_reviews, brand)

        # Summary
        print(f"\n  Summary for {brand}:")
        df = pd.DataFrame(all_reviews)
        if "rating" in df.columns:
            print(f"    Total: {len(df)}")
            print(f"    Avg rating: {df['rating'].mean():.1f}")
            print(f"    Rating dist: {df['rating'].value_counts().sort_index().to_dict()}")
            has_text = len(df[df["text_original"].fillna("").str.len() > 10])
            print(f"    With text (>10 chars): {has_text}")


def main():
    parser = argparse.ArgumentParser(description="Collect app store reviews")
    parser.add_argument("--discover", action="store_true", help="Discover app IDs")
    parser.add_argument("--store", default=None, choices=["google_play", "app_store"])
    parser.add_argument("--count", type=int, default=200, help="Max reviews per store")
    args = parser.parse_args()

    if args.discover:
        discover_apps()
    else:
        collect(store_filter=args.store, count=args.count)


if __name__ == "__main__":
    main()
