"""Collect VoC from insurance-specific communities.

Targets foreign insurer comparison discussions and MetLife-specific product reviews.
Uses existing Naver Search API with specialized keywords.

Usage:
    python scripts/collect_foreign_voc.py
    python scripts/collect_foreign_voc.py --max-per-query 200
"""

import sys
import os
import re
import time
import argparse
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
import pandas as pd
from dotenv import load_dotenv
from config.db import get_conn, _use_csv_fallback

load_dotenv()

CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

HEADERS = {
    "X-Naver-Client-Id": CLIENT_ID,
    "X-Naver-Client-Secret": CLIENT_SECRET,
}

# Foreign insurer focused keywords
FOREIGN_KEYWORDS = [
    # MetLife vs competitors
    "외국계 보험 비교",
    "메트라이프 AIG 비교",
    "메트라이프 푸르덴셜 비교",
    "메트라이프 처브 비교",
    "외국계 생명보험 추천",
    "외국계 보험 장단점",
    # MetLife product-specific
    "메트라이프 달러보험 후기",
    "메트라이프 변액보험 후기",
    "메트라이프 종신보험 후기",
    "메트라이프 연금보험 후기",
    "메트라이프 360Health",
    "메트라이프원 앱 후기",
    # Insurance expert community topics
    "보험 설계사 메트라이프",
    "메트라이프 프로지점",
    "외국계 보험 해약",
    "외국계 보험 민원",
]

SOURCES = ["blog", "cafearticle"]

BRAND_PATTERNS = {
    "메트라이프": "metlife",
    "메트라이프생명": "metlife",
    "AIG": "aig",
    "에이아이지": "aig",
    "푸르덴셜": "prudential",
    "처브": "chubb",
    "삼성생명": "samsung",
    "한화생명": "hanwha",
    "교보생명": "kyobo",
}


def search_naver(query, source, display=100, start=1):
    url = f"https://openapi.naver.com/v1/search/{source}.json"
    params = {"query": query, "display": display, "start": start, "sort": "date"}
    resp = requests.get(url, headers=HEADERS, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def clean_html(text):
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", text)


def detect_brands(text):
    """Detect all brands mentioned in text (can be multiple)."""
    found = set()
    for pattern, brand_key in BRAND_PATTERNS.items():
        if pattern.lower() in text.lower():
            found.add(brand_key)
    return list(found) if found else ["general"]


def collect_source(source, max_per_query=200):
    if not CLIENT_ID or not CLIENT_SECRET:
        print("[ERROR] NAVER_CLIENT_ID / NAVER_CLIENT_SECRET not set")
        return []

    all_items = []

    for query in FOREIGN_KEYWORDS:
        print(f"  Query: '{query}' ({source})")
        start = 1
        collected = 0

        while collected < max_per_query:
            try:
                data = search_naver(
                    query, source,
                    display=min(100, max_per_query - collected),
                    start=start,
                )
            except Exception as e:
                print(f"    [WARN] API error: {e}")
                break

            items = data.get("items", [])
            if not items:
                break

            for item in items:
                title = clean_html(item.get("title", ""))
                description = clean_html(item.get("description", ""))
                full_text = f"{title} {description}"

                brands = detect_brands(full_text)
                if not any(b in brands for b in ["metlife"]):
                    brands_from_query = detect_brands(query)
                    brands = list(set(brands + brands_from_query))

                primary_brand = "metlife" if "metlife" in brands else brands[0]

                postdate = item.get("postdate", "")
                if postdate:
                    try:
                        postdate = datetime.strptime(postdate, "%Y%m%d").date()
                    except ValueError:
                        postdate = None
                else:
                    postdate = None

                all_items.append({
                    "source": f"foreign_{source}",
                    "title": title,
                    "description": description,
                    "link": item.get("link", ""),
                    "blogger_name": item.get("bloggername", item.get("cafename", "")),
                    "postdate": postdate,
                    "brand": primary_brand,
                    "brands_mentioned": ",".join(brands),
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
    seen = set()
    unique = []
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
                try:
                    cur.execute(
                        """
                        INSERT INTO raw.naver_voc
                            (source, title, description, link, blogger_name,
                             postdate, brand, query)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            item["source"], item["title"], item["description"],
                            item["link"], item["blogger_name"],
                            item["postdate"], item["brand"], item["query"],
                        ),
                    )
                    inserted += 1
                except Exception:
                    pass
    print(f"  DB: {inserted} rows inserted")


def save_to_csv(items, path="data/raw/foreign_voc.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(items)
    if os.path.exists(path):
        existing = pd.read_csv(path)
        df = pd.concat([existing, df]).drop_duplicates(subset=["link"], keep="last")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  CSV: {len(df)} rows saved to {path}")


def run(max_per_query=200):
    all_items = []
    for source in SOURCES:
        print(f"\nCollecting foreign insurer VoC from: {source}")
        items = collect_source(source, max_per_query=max_per_query)
        all_items.extend(items)

    unique_items = deduplicate(all_items)
    print(f"\nTotal: {len(all_items)} raw -> {len(unique_items)} unique")

    # Brand distribution
    brand_counts = {}
    for item in unique_items:
        b = item["brand"]
        brand_counts[b] = brand_counts.get(b, 0) + 1
    print("Brand distribution:")
    for brand, count in sorted(brand_counts.items(), key=lambda x: -x[1]):
        print(f"  {brand}: {count}")

    # Brands co-mention analysis
    comention = {}
    for item in unique_items:
        brands = item.get("brands_mentioned", "").split(",")
        if len(brands) > 1:
            key = tuple(sorted(brands))
            comention[key] = comention.get(key, 0) + 1
    if comention:
        print("\nBrand co-mentions (top 10):")
        for pair, count in sorted(comention.items(), key=lambda x: -x[1])[:10]:
            print(f"  {' vs '.join(pair)}: {count}")

    # Save
    if _use_csv_fallback():
        save_to_csv(unique_items)
    else:
        save_to_db(unique_items)
        save_to_csv(unique_items)

    return unique_items


def main():
    parser = argparse.ArgumentParser(description="Collect foreign insurer VoC")
    parser.add_argument("--max-per-query", type=int, default=200)
    args = parser.parse_args()
    run(max_per_query=args.max_per_query)


if __name__ == "__main__":
    main()
