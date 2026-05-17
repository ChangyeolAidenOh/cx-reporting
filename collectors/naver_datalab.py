"""Collector: Naver DataLab brand search trends (weekly).

Fetches relative search volume for 4 insurance brands from Naver DataLab API.
Stores results in raw.naver_datalab (PostgreSQL) or CSV fallback.

Usage:
    python -m collectors.naver_datalab
    python -m collectors.naver_datalab --start 2024-01-01 --end 2026-05-14
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import argparse
import json
from datetime import datetime

import requests
import pandas as pd
from dotenv import load_dotenv

from config.settings import BRANDS, NAVER_DATALAB
from config.db import get_conn, _use_csv_fallback

load_dotenv()

CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

HEADERS = {
    "X-Naver-Client-Id": CLIENT_ID,
    "X-Naver-Client-Secret": CLIENT_SECRET,
    "Content-Type": "application/json",
}

API_URL = "https://openapi.naver.com/v1/datalab/search"


def build_keyword_groups():
    """Build keyword groups from settings."""
    groups = []
    for brand_key, brand_info in BRANDS.items():
        groups.append({
            "groupName": brand_info["name_kr"],
            "keywords": brand_info["naver_keywords"],
        })
    return groups


def fetch_trends(start_date, end_date, time_unit="week"):
    """Fetch search trend data from Naver DataLab API."""
    body = {
        "startDate": start_date,
        "endDate": end_date,
        "timeUnit": time_unit,
        "keywordGroups": build_keyword_groups(),
    }

    resp = requests.post(API_URL, headers=HEADERS, json=body, timeout=15)
    resp.raise_for_status()
    return resp.json()


def parse_response(response):
    """Parse API response into a DataFrame."""
    rows = []
    for group in response.get("results", []):
        brand = group["title"]
        for point in group.get("data", []):
            rows.append({
                "brand": brand,
                "period": point["period"],
                "ratio": point["ratio"],
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["period"] = pd.to_datetime(df["period"])
    return df


def save_to_db(df):
    """Insert or update records in raw.naver_datalab."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            for _, row in df.iterrows():
                cur.execute(
                    """
                    INSERT INTO raw.naver_datalab (brand, period, ratio)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (brand, period)
                    DO UPDATE SET ratio = EXCLUDED.ratio,
                                  collected_at = NOW()
                    """,
                    (row["brand"], row["period"].date(), float(row["ratio"])),
                )
    print(f"  DB: {len(df)} rows upserted to raw.naver_datalab")


def save_to_csv(df, path="data/raw/naver_datalab.csv"):
    """Save to CSV (append mode)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        existing = pd.read_csv(path, parse_dates=["period"])
        df = pd.concat([existing, df]).drop_duplicates(
            subset=["brand", "period"], keep="last"
        )

    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  CSV: {len(df)} rows saved to {path}")


def collect(start_date=None, end_date=None):
    """Main collection entry point."""
    if not CLIENT_ID or not CLIENT_SECRET:
        print("[ERROR] NAVER_CLIENT_ID / NAVER_CLIENT_SECRET not set")
        return None

    start = start_date or NAVER_DATALAB["start_date"]
    end = end_date or datetime.now().strftime("%Y-%m-%d")

    print(f"Collecting Naver DataLab trends: {start} ~ {end}")
    response = fetch_trends(start, end)
    df = parse_response(response)

    if df.empty:
        print("[WARN] No data returned")
        return None

    print(f"  Fetched {len(df)} data points for {df['brand'].nunique()} brands")

    if _use_csv_fallback():
        save_to_csv(df)
    else:
        save_to_db(df)
        save_to_csv(df)  # always keep CSV copy

    return df


def main():
    parser = argparse.ArgumentParser(description="Collect Naver DataLab trends")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    collect(start_date=args.start, end_date=args.end)


if __name__ == "__main__":
    main()
