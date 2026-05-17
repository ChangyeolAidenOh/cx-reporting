"""Process app reviews: staging insert + sentiment analysis."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from config.db import get_conn
from etl.staging.text_preprocessing import clean_text
from analysis.sentiment import dict_sentiment, score_to_label, call_haiku_batch


def run():
    # Extract
    with get_conn() as conn:
        df = pd.read_sql(
            "SELECT id, app_id, store, brand, rating, text_original, posted_at "
            "FROM raw.app_reviews WHERE LENGTH(text_original) >= 10",
            conn,
        )
    print(f"Loaded {len(df)} app reviews")

    # Clean and insert to staging
    with get_conn() as conn:
        with conn.cursor() as cur:
            inserted = 0
            for _, row in df.iterrows():
                text_clean = clean_text(row["text_original"])
                if len(text_clean) < 10:
                    continue

                # Sentiment via dictionary
                score, confidence, matched = dict_sentiment(text_clean)
                label = score_to_label(score, confidence)

                # For app reviews, rating itself is a strong signal
                if confidence < 0.3:
                    if row["rating"] <= 2:
                        score = -0.7
                        label = "negative"
                    elif row["rating"] >= 4:
                        score = 0.7
                        label = "positive"
                    else:
                        score = 0.0
                        label = "neutral"
                    method = "rating_fallback"
                else:
                    method = "dictionary"

                cur.execute(
                    """
                    INSERT INTO staging.content_enriched
                        (source, source_id, brand, title, text_clean,
                         topic_id, topic_label, sentiment_score,
                         sentiment_label, sentiment_method, published_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        f"app_{row['store']}", str(row["id"]), row["brand"],
                        f"[{row['rating']}★] App Review", text_clean,
                        -2, "app_review",  # separate topic category
                        float(score), label, method,
                        row["posted_at"] if pd.notna(row["posted_at"]) else None,
                    ),
                )
                inserted += 1
    print(f"Inserted {inserted} reviews to staging.content_enriched")

    # Summary
    with get_conn() as conn:
        summary = pd.read_sql(
            """
            SELECT sentiment_label, sentiment_method, COUNT(*) as cnt
            FROM staging.content_enriched
            WHERE topic_label = 'app_review'
            GROUP BY sentiment_label, sentiment_method
            ORDER BY cnt DESC
            """,
            conn,
        )
    print(f"\nApp review sentiment summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run()