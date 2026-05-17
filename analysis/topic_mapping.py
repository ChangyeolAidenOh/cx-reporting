"""Apply manual topic category mapping to staging.content_enriched.

Maps 43 BERTopic topics -> 6 high-level categories.

Usage:
    python -m analysis.topic_mapping
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.db import get_conn

# BERTopic ID -> high-level category
TOPIC_CATEGORY = {
    # 상품홍보
    0: "상품홍보", 6: "상품홍보", 21: "상품홍보",
    24: "상품홍보", 36: "상품홍보", 40: "상품홍보",
    # 교육·전문성
    4: "교육·전문성", 5: "교육·전문성", 9: "교육·전문성",
    10: "교육·전문성", 14: "교육·전문성", 15: "교육·전문성",
    23: "교육·전문성", 41: "교육·전문성", 16: "교육·전문성",
    37: "교육·전문성",
    # 건강·라이프
    2: "건강·라이프", 3: "건강·라이프", 13: "건강·라이프",
    19: "건강·라이프", 20: "건강·라이프", 31: "건강·라이프",
    12: "건강·라이프", 39: "건강·라이프",
    # 이벤트·프로모션
    8: "이벤트·프로모션", 11: "이벤트·프로모션", 18: "이벤트·프로모션",
    26: "이벤트·프로모션", 27: "이벤트·프로모션", 35: "이벤트·프로모션",
    30: "이벤트·프로모션", 34: "이벤트·프로모션",
    # 고객후기
    1: "고객후기", 17: "고객후기", 28: "고객후기",
    33: "고객후기", 38: "고객후기", 42: "고객후기",
    # noise
    -1: "noise", 7: "noise", 22: "noise",
    25: "noise", 29: "noise", 32: "noise",

}


def run():
    with get_conn() as conn:
        with conn.cursor() as cur:
            for topic_id, category in TOPIC_CATEGORY.items():
                cur.execute(
                    """
                    UPDATE staging.content_enriched
                    SET topic_label = %s
                    WHERE topic_id = %s
                    """,
                    (category, topic_id),
                )

            # Verify
            cur.execute(
                """
                SELECT topic_label, COUNT(*) as cnt
                FROM staging.content_enriched
                GROUP BY topic_label
                ORDER BY cnt DESC
                """
            )
            results = cur.fetchall()

    print("Topic category distribution:")
    for label, count in results:
        print(f"  {label}: {count}")


if __name__ == "__main__":
    run()