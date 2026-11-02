"""ETL: raw VoC + YouTube comments -> staging.content_enriched (text prep)

Cleans and normalizes text for downstream NLP (BERTopic, sentiment).
Topic and sentiment columns remain NULL -- populated in the NLP stage.
Brand names are unified to settings brand keys.

Usage:
    python -m etl.staging.text_preprocessing
"""

import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
import pandas as pd
from config.db import get_conn
from config.settings import BRAND_KR_TO_KEY


def clean_text(text):
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"&#\d+;", " ", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    text = re.sub(r"[^\w\sㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9.,!?~·\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def to_brand_key(brand):
    return BRAND_KR_TO_KEY.get(brand, brand)


def extract_voc():
    with get_conn() as conn:
        df = pd.read_sql(
            "SELECT id, source, title, description, brand, postdate FROM raw.naver_voc", conn)
    print(f"Extracted {len(df)} VoC rows")
    return df


def extract_youtube_comments():
    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT c.id, c.comment_id, c.video_id, c.text_original, c.published_at, v.brand
            FROM raw.youtube_comments c
            JOIN raw.youtube_videos v ON c.video_id = v.video_id
            """, conn)
    print(f"Extracted {len(df)} YouTube comments")
    return df


def transform_voc(df):
    records = []
    for _, row in df.iterrows():
        title_clean = clean_text(row["title"])
        desc_clean = clean_text(row["description"])
        combined = f"{title_clean} {desc_clean}".strip()
        if len(combined) < 10:
            continue
        records.append({
            "source": row["source"],
            "source_id": str(row["id"]),
            "brand": to_brand_key(row["brand"]),
            "title": title_clean,
            "text_clean": combined,
            "published_at": row["postdate"],
        })
    result = pd.DataFrame(records)
    print(f"VoC: {len(df)} raw -> {len(result)} after cleaning (dropped {len(df) - len(result)} short)")
    return result


def transform_comments(df):
    records = []
    for _, row in df.iterrows():
        text_clean = clean_text(row["text_original"])
        if len(text_clean) < 10:
            continue
        records.append({
            "source": "youtube_comment",
            "source_id": row["comment_id"],
            "brand": to_brand_key(row["brand"]),
            "title": "",
            "text_clean": text_clean,
            "published_at": row["published_at"],
        })
    result = pd.DataFrame(records)
    print(f"Comments: {len(df)} raw -> {len(result)} after cleaning (dropped {len(df) - len(result)} short)")
    return result


def load(df):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM staging.content_enriched")
            for _, row in df.iterrows():
                cur.execute(
                    """
                    INSERT INTO staging.content_enriched
                        (source, source_id, brand, title, text_clean, published_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (row["source"], row["source_id"], row["brand"], row["title"],
                     row["text_clean"],
                     row["published_at"] if pd.notna(row.get("published_at")) else None),
                )
    print(f"Loaded {len(df)} rows to staging.content_enriched")


def save_csv(df, path="data/processed/content_enriched_text.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"CSV exported: {path}")


def run():
    voc_clean = transform_voc(extract_voc())
    comments_clean = transform_comments(extract_youtube_comments())
    combined = pd.concat([voc_clean, comments_clean], ignore_index=True)
    print(f"\nCombined: {len(combined)} documents")
    load(combined)
    save_csv(combined)
    print("\nStaging summary:")
    print(f"  Sources: {combined['source'].value_counts().to_dict()}")
    print(f"  Brands: {combined['brand'].value_counts().to_dict()}")
    print(f"  Avg text length: {combined['text_clean'].str.len().mean():.0f} chars")


if __name__ == "__main__":
    run()
