"""Split 태아·어린이보험 out of 상품 설계·고객후기 by rule.

Why: in the blog+cafe customer segment ~41% of 상품 설계·고객후기 docs are about
태아보험/어린이보험 (82% for 현대해상), so the category mixed a product line
into a discourse axis and produced a spurious 현대해상 over-index.
A regex rule is explainable and precise (38/40 in sample review); leakage
into other categories is near zero, so the rule is scoped to
topic_label = '상품 설계·고객후기' only.

Rows get label_source = 'rule_fetal' so analysis/topic_mapping.py skips them
(it only rewrites rows with COALESCE(label_source, 'mapping') = 'mapping').

Usage:
    python -m scripts.relabel_fetal_child            dry-run: counts + samples
    python -m scripts.relabel_fetal_child --apply    write to staging
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from config.db import get_conn

SOURCE_LABEL = "상품 설계·고객후기"
NEW_LABEL = "태아·어린이보험"
LABEL_SOURCE = "rule_fetal"
PATTERN = r"태아보험|태아 보험|어린이보험|어린이 보험|자녀보험|태아특약|산모특약|태아 ?종합|어린이 ?종합보험"

MATCH_SQL = f"""
    topic_label = %s
    AND (COALESCE(title, '') || ' ' || text_clean) ~ %s
"""


def dry_run():
    with get_conn() as conn:
        summary = pd.read_sql(f"""
            SELECT source, author_type, COUNT(*) AS n
            FROM staging.content_enriched
            WHERE {MATCH_SQL}
            GROUP BY 1, 2 ORDER BY 1, 2""", conn, params=(SOURCE_LABEL, PATTERN))
        by_brand = pd.read_sql(f"""
            SELECT brand, COUNT(*) AS n
            FROM staging.content_enriched
            WHERE {MATCH_SQL}
            GROUP BY 1 ORDER BY 1""", conn, params=(SOURCE_LABEL, PATTERN))
        already = pd.read_sql("""
            SELECT COUNT(*) AS n FROM staging.content_enriched
            WHERE topic_label = %s""", conn, params=(NEW_LABEL,))["n"][0]
        advisor_sample = pd.read_sql(f"""
            SELECT id, brand, LEFT(title, 50) AS title, LEFT(text_clean, 150) AS text
            FROM staging.content_enriched
            WHERE {MATCH_SQL} AND author_type = 'advisor'
            ORDER BY md5(id::text) LIMIT 20""", conn, params=(SOURCE_LABEL, PATTERN))

    print(f"Docs to relabel {SOURCE_LABEL} -> {NEW_LABEL}: {int(summary['n'].sum())}")
    print(f"Already labeled {NEW_LABEL}: {int(already)}")
    print("\nBy source x author_type:")
    print(summary.to_string(index=False))
    print("\nBy brand:")
    print(by_brand.to_string(index=False))
    print("\nAdvisor sample (20):")
    for _, r in advisor_sample.iterrows():
        print(f"[{r['id']}][{r['brand']}] {r['title']}\n    {r['text']}\n")


def apply():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                UPDATE staging.content_enriched
                SET topic_label = %s, label_source = %s, processed_at = NOW()
                WHERE {MATCH_SQL}""", (NEW_LABEL, LABEL_SOURCE, SOURCE_LABEL, PATTERN))
            n = cur.rowcount
        conn.commit()
    print(f"Relabeled {n} docs to {NEW_LABEL} (label_source={LABEL_SOURCE})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    if args.apply:
        apply()
    else:
        dry_run()
