"""ETL: staging.content_enriched -> mart.voc_sentiment

Weekly sentiment aggregation per brand.

Usage:
    python -m etl.mart.voc_sentiment
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
import json
import pandas as pd
from config.db import get_conn


def extract():
    """Load enriched content with sentiment."""
    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT brand, published_at, sentiment_label, text_clean, topic_label
            FROM staging.content_enriched
            WHERE sentiment_label IS NOT NULL
              AND topic_label != 'noise'
              AND published_at IS NOT NULL
            """,
            conn,
        )
    df["published_at"] = pd.to_datetime(df["published_at"])
    print(f"Extracted {len(df)} docs with sentiment + date")
    return df


def extract_keywords(texts, top_n=5):
    """Extract top keywords from texts using simple frequency."""
    from collections import Counter
    import re

    words = []
    for text in texts:
        if isinstance(text, str):
            tokens = re.findall(r"[가-힣]{2,}", text)
            words.extend(tokens)

    # Filter common stopwords
    stopwords = {"보험", "가입", "상품", "상담", "정보", "확인", "관련",
                 "경우", "이용", "안내", "사항", "제공", "부분", "정도"}
    words = [w for w in words if w not in stopwords]

    return [w for w, _ in Counter(words).most_common(top_n)]


def transform(df):
    """Compute weekly sentiment ratios per brand."""
    df["week_start"] = df["published_at"].dt.to_period("W-SUN").apply(
        lambda x: x.start_time
    )

    records = []
    for (brand, week), group in df.groupby(["brand", "week_start"]):
        total = len(group)
        pos = len(group[group["sentiment_label"] == "positive"])
        neu = len(group[group["sentiment_label"] == "neutral"])
        neg = len(group[group["sentiment_label"] == "negative"])

        keywords = extract_keywords(group["text_clean"].tolist())

        records.append({
            "brand": brand,
            "week_start": week,
            "positive_ratio": round(pos / total, 3) if total > 0 else 0,
            "neutral_ratio": round(neu / total, 3) if total > 0 else 0,
            "negative_ratio": round(neg / total, 3) if total > 0 else 0,
            "total_docs": total,
            "top_keywords": json.dumps(keywords, ensure_ascii=False),
        })

    result = pd.DataFrame(records)
    print(f"Transformed to {len(result)} weekly records")
    return result


def load(df):
    """Upsert into mart.voc_sentiment."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            for _, row in df.iterrows():
                cur.execute(
                    """
                    INSERT INTO mart.voc_sentiment
                        (brand, week_start, positive_ratio, neutral_ratio,
                         negative_ratio, total_docs, top_keywords)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (brand, week_start)
                    DO UPDATE SET positive_ratio = EXCLUDED.positive_ratio,
                                  neutral_ratio = EXCLUDED.neutral_ratio,
                                  negative_ratio = EXCLUDED.negative_ratio,
                                  total_docs = EXCLUDED.total_docs,
                                  top_keywords = EXCLUDED.top_keywords
                    """,
                    (
                        row["brand"], row["week_start"].date(),
                        float(row["positive_ratio"]), float(row["neutral_ratio"]),
                        float(row["negative_ratio"]), int(row["total_docs"]),
                        row["top_keywords"],
                    ),
                )
    print(f"Loaded {len(df)} rows to mart.voc_sentiment")


def save_csv(df, path="data/exports/voc_sentiment.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"CSV exported: {path}")


def run():
    df = extract()
    result = transform(df)
    load(result)
    save_csv(result)

    print("\nSentiment summary by brand:")
    for brand in sorted(result["brand"].unique()):
        bdf = result[result["brand"] == brand]
        avg_pos = bdf["positive_ratio"].mean()
        avg_neg = bdf["negative_ratio"].mean()
        print(f"  {brand}: avg positive {avg_pos:.1%}, avg negative {avg_neg:.1%}, "
              f"{bdf['total_docs'].sum()} total docs across {len(bdf)} weeks")


if __name__ == "__main__":
    run()
