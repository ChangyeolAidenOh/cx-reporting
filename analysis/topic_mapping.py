"""Apply manual topic category mapping to staging.content_enriched.

Maps BERTopic stage-1 (0-58) and stage-2 (100-109, 200-214) topics
-> 7 business categories for 손해보험 content analysis.

Usage:
    python -m analysis.topic_mapping
"""
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.db import get_conn

CATEGORY_TOPICS = {
    "자동차보험": [201, 202, 211, 212, 214, 102],
    "실손·건강보험": [204, 208, 209, 210, 109, 105],
    "보험금 청구·보상": [205, 207],
    "상품 설계·고객후기": [200, 203, 206, 213, 108, 101, 0, 100],
    "브랜드 캠페인": [1, 2, 5, 7, 15, 16, 18, 19, 21, 22, 26, 30, 31, 36, 39, 54,
                11, 14, 17, 20, 27, 34, 37, 46, 55, 57, 58],
    "이벤트·프로모션": [6, 10, 12, 13, 24, 28, 29, 35, 38, 41, 48, 49, 50,
                 3, 8, 23, 47, 52],
    "noise": [-1, 4, 9, 25, 32, 33, 40, 42, 43, 44, 45, 51, 53, 56,
              103, 104, 106, 107],
}

# Flatten to topic_id -> category
TOPIC_CATEGORY = {t: cat for cat, ids in CATEGORY_TOPICS.items() for t in ids}


def run():
    with get_conn() as conn:
        with conn.cursor() as cur:
            for topic_id, category in TOPIC_CATEGORY.items():
                cur.execute(
                    "UPDATE staging.content_enriched SET topic_label = %s "
                    "WHERE topic_id = %s AND COALESCE(label_source, 'mapping') = 'mapping'",
                    (category, topic_id),
                )

            # Any topic id not in the mapping -> flag so it is not silently dropped
            cur.execute(
                """
                SELECT topic_id, COUNT(*) FROM staging.content_enriched
                WHERE topic_id NOT IN %s GROUP BY topic_id
                """,
                (tuple(TOPIC_CATEGORY.keys()),),
            )
            unmapped = cur.fetchall()

            cur.execute(
                """
                SELECT topic_label, COUNT(*) AS cnt
                FROM staging.content_enriched
                GROUP BY topic_label ORDER BY cnt DESC
                """
            )
            results = cur.fetchall()

    print("Topic category distribution:")
    total = sum(c for _, c in results)
    for label, count in results:
        print(f"  {label}: {count} ({count / total * 100:.1f}%)")

    if unmapped:
        print("\n[WARN] Unmapped topic ids (still carry raw BERTopic label):")
        for t, c in unmapped:
            print(f"  topic {t}: {c} docs")

    with get_conn() as conn:
        pd.read_sql("SELECT id, brand, source, topic_id, topic_label, sentiment_label, sentiment_score "
                    "FROM staging.content_enriched", conn).to_csv(
            "data/exports/content_enriched.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    run()
