"""Process foreign insurer VoC: staging insert + sentiment analysis."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from config.db import get_conn
from etl.staging.text_preprocessing import clean_text
from analysis.sentiment import dict_sentiment, score_to_label


def run():
    # Extract only foreign VoC (not yet in staging)
    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT id, source, title, description, brand, postdate
            FROM raw.naver_voc
            WHERE source IN ('foreign_blog', 'foreign_cafearticle')
            """,
            conn,
        )
    print(f"Loaded {len(df)} foreign VoC rows")

    # Clean, score, insert
    with get_conn() as conn:
        with conn.cursor() as cur:
            inserted = 0
            for _, row in df.iterrows():
                title_clean = clean_text(row["title"])
                desc_clean = clean_text(row["description"])
                combined = f"{title_clean} {desc_clean}".strip()

                if len(combined) < 10:
                    continue

                # Sentiment
                score, confidence, matched = dict_sentiment(combined)
                label = score_to_label(score, confidence)
                method = "dictionary" if confidence >= 0.3 else "low_conf_dict"

                cur.execute(
                    """
                    INSERT INTO staging.content_enriched
                        (source, source_id, brand, title, text_clean,
                         topic_id, topic_label, sentiment_score,
                         sentiment_label, sentiment_method, published_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        row["source"], str(row["id"]), row["brand"],
                        title_clean, combined,
                        -3, "foreign_comparison",
                        float(score), label, method,
                        row["postdate"] if pd.notna(row["postdate"]) else None,
                    ),
                )
                inserted += 1
    print(f"Inserted {inserted} rows to staging.content_enriched")

    # Summary
    with get_conn() as conn:
        summary = pd.read_sql(
            """
            SELECT sentiment_label, COUNT(*) as cnt
            FROM staging.content_enriched
            WHERE topic_label = 'foreign_comparison'
            GROUP BY sentiment_label
            ORDER BY cnt DESC
            """,
            conn,
        )
    print(f"\nForeign VoC sentiment summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run()