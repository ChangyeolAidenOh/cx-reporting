"""Collector: Naver Blog/Cafe VoC (consumer voice).

Fetches blog posts and cafe articles related to insurance brands
from Naver Search API. Stores in raw.naver_voc.

Note: Cafe search results do not include postdate in API response.

Usage:
    python -m collectors.naver_voc
    python -m collectors.naver_voc --source blog --max-per-query 500
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import re
import argparse
import time
from datetime import datetime

import requests
import pandas as pd
from dotenv import load_dotenv

from config.settings import BRANDS, VOC_KEYWORDS
from config.db import get_conn, _use_csv_fallback

load_dotenv()

CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

HEADERS = {
    "X-Naver-Client-Id": CLIENT_ID,
    "X-Naver-Client-Secret": CLIENT_SECRET,
}

SOURCES = ["blog", "cafearticle"]

# Brand detection patterns derived from settings (all keywords per brand)
BRAND_PATTERNS = {
    kw: key
    for key, info in BRANDS.items()
    for kw in [info["name_kr"]] + info["naver_keywords"]
}


def search_naver(query, source, display=100, start=1):
    url = f"https://openapi.naver.com/v1/search/{source}.json"
    params = {"query": query, "display": display, "start": start, "sort": "date"}
    resp = requests.get(url, headers=HEADERS, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def detect_brand(text):
    """Detect brand from text; 'general' if no brand mentioned."""
    text_lower = text.lower()
    for pattern, brand_key in BRAND_PATTERNS.items():
        if pattern.lower() in text_lower:
            return brand_key
    return "general"


def clean_html(text):
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", text)


def collect_source(source, max_per_query=200):
    if not CLIENT_ID or not CLIENT_SECRET:
        print("[ERROR] NAVER_CLIENT_ID / NAVER_CLIENT_SECRET not set")
        return []

    all_items = []
    for query in VOC_KEYWORDS:
        print(f"  Query: '{query}' ({source})")
        start, collected = 1, 0

        while collected < max_per_query:
            try:
                data = search_naver(query, source,
                                    display=min(100, max_per_query - collected), start=start)
            except Exception as e:
                print(f"    [WARN] API error at start={start}: {e}")
                break

            items = data.get("items", [])
            if not items:
                break

            for item in items:
                title = clean_html(item.get("title", ""))
                description = clean_html(item.get("description", ""))
                brand = detect_brand(f"{title} {description}")
                if brand == "general":
                    brand = detect_brand(query)

                postdate = item.get("postdate", "")
                try:
                    postdate = datetime.strptime(postdate, "%Y%m%d").date() if postdate else None
                except ValueError:
                    postdate = None

                all_items.append({
                    "source": source,
                    "title": title,
                    "description": description,
                    "link": item.get("link", ""),
                    "blogger_name": item.get("bloggername", item.get("cafename", "")),
                    "postdate": postdate,
                    "brand": brand,
                    "query": query,
                })

            collected += len(items)
            start += len(items)
            if start > 1000:
                break
            time.sleep(0.1)

        print(f"    Collected {collected} items")
    return all_items


def deduplicate(items):
    seen, unique = set(), []
    for item in items:
        link = item.get("link", "")
        if link and link not in seen:
            seen.add(link)
            unique.append(item)
    return unique


def save_to_db(items):
    with get_conn() as conn:
        with conn.cursor() as cur:
            inserted = 0
            for item in items:
                cur.execute(
                    """
                    INSERT INTO raw.naver_voc
                        (source, title, description, link, blogger_name, postdate, brand, query)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (item["source"], item["title"], item["description"], item["link"],
                     item["blogger_name"], item["postdate"], item["brand"], item["query"]),
                )
                inserted += 1
    print(f"  DB: {inserted} rows inserted to raw.naver_voc")


def save_to_csv(items, path="data/raw/naver_voc.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(items)
    if os.path.exists(path):
        existing = pd.read_csv(path)
        df = pd.concat([existing, df]).drop_duplicates(subset=["link"], keep="last")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  CSV: {len(df)} rows saved to {path}")


def collect(source_filter=None, max_per_query=200):
    sources = [source_filter] if source_filter else SOURCES
    all_items = []
    for source in sources:
        print(f"\nCollecting VoC from: {source}")
        all_items.extend(collect_source(source, max_per_query=max_per_query))

    unique_items = deduplicate(all_items)
    print(f"\nTotal: {len(all_items)} raw -> {len(unique_items)} unique")

    brand_counts = {}
    for item in unique_items:
        brand_counts[item["brand"]] = brand_counts.get(item["brand"], 0) + 1
    print("Brand distribution:")
    for brand, count in sorted(brand_counts.items(), key=lambda x: -x[1]):
        print(f"  {brand}: {count}")

    if _use_csv_fallback():
        save_to_csv(unique_items)
    else:
        save_to_db(unique_items)
        save_to_csv(unique_items)
    return unique_items


def main():
    parser = argparse.ArgumentParser(description="Collect Naver Blog/Cafe VoC")
    parser.add_argument("--source", default=None, choices=["blog", "cafearticle"])
    parser.add_argument("--max-per-query", type=int, default=200)
    args = parser.parse_args()
    collect(source_filter=args.source, max_per_query=args.max_per_query)


if __name__ == "__main__":
    main()
