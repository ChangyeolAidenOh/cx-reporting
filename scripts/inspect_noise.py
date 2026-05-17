"""Inspect noise-labeled documents for potential reclassification.

Usage:
    python scripts/inspect_noise.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from config.db import get_conn


def run():
    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT topic_id, brand, source, text_clean
            FROM staging.content_enriched
            WHERE topic_label = 'noise'
            ORDER BY topic_id
            """,
            conn,
        )

    print(f"Total noise docs: {len(df)}\n")

    for topic_id in sorted(df["topic_id"].unique()):
        tdf = df[df["topic_id"] == topic_id]
        print(f"{'=' * 70}")
        print(f"Topic {topic_id} ({len(tdf)} docs)")
        print(f"  Brands: {tdf['brand'].value_counts().to_dict()}")
        print(f"  Sources: {tdf['source'].value_counts().to_dict()}")
        print(f"  Samples:")
        for _, row in tdf.sample(min(5, len(tdf)), random_state=42).iterrows():
            text = row["text_clean"][:120]
            print(f"    [{row['brand']}] {text}")
        print()


if __name__ == "__main__":
    run()