"""Backfill author_type for staging docs with Claude (resumable).

Prompt, parser and retry live in analysis/author_llm.py and are shared with the
weekly pipeline (orchestration/incremental.py), so a prompt change applies to both.

Usage:
    python -m scripts.label_author_type --limit 100        # smoke test
    python -m scripts.label_author_type                    # label all NULL blog/cafe rows (resumable)
    python -m scripts.label_author_type --verify           # Sonnet re-check of Haiku 'customer' labels
    python -m scripts.label_author_type --sample 40        # print a random check sample
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from dotenv import load_dotenv
from config.db import get_conn
from analysis.author_llm import label_author_docs, verify_customer_docs, HAIKU_MODEL, SONNET_MODEL

load_dotenv(".env")


def ensure_column():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE staging.content_enriched ADD COLUMN IF NOT EXISTS author_type VARCHAR(20)")
            cur.execute("ALTER TABLE staging.content_enriched ADD COLUMN IF NOT EXISTS author_type_src VARCHAR(20)")


def fetch_unlabeled(limit=None):
    q = ("SELECT id, text_clean FROM staging.content_enriched "
         "WHERE author_type IS NULL AND source IN ('blog','cafearticle') ORDER BY id")
    if limit:
        q += f" LIMIT {int(limit)}"
    with get_conn() as conn:
        return pd.read_sql(q, conn)


def apply_source_rules():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE staging.content_enriched SET author_type='customer', author_type_src='rule' "
                        "WHERE source='youtube_comment' AND author_type IS NULL")
            n = cur.rowcount
    print(f"[rule] youtube_comment -> customer: {n}")


def save(labels, src):
    if not labels:
        return
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany("UPDATE staging.content_enriched SET author_type=%s, author_type_src=%s WHERE id=%s",
                            [(t, src, int(i)) for i, t in labels.items()])


def run(limit=None):
    ensure_column()
    apply_source_rules()
    df = fetch_unlabeled(limit)
    if df.empty:
        print("nothing to label"); return
    print(f"Labeling {len(df)} docs with {HAIKU_MODEL}")
    labels = label_author_docs(df)
    save(labels, "haiku")
    print(f"\nDone: {len(labels)} labeled, {len(df) - len(labels)} failed (re-run to retry)")
    with get_conn() as conn:
        dist = pd.read_sql("SELECT author_type, COUNT(*) n FROM staging.content_enriched "
                           "GROUP BY 1 ORDER BY 2 DESC", conn)
    print(dist.to_string(index=False))


def verify(limit=None):
    q = ("SELECT id, text_clean FROM staging.content_enriched "
         "WHERE author_type='customer' AND COALESCE(author_type_src,'haiku')='haiku' ORDER BY id")
    if limit:
        q += f" LIMIT {int(limit)}"
    with get_conn() as conn:
        df = pd.read_sql(q, conn)
    if df.empty:
        print("nothing to verify"); return
    print(f"Verifying {len(df)} customer labels with {SONNET_MODEL}")
    out = verify_customer_docs(df)
    save(out, "sonnet")
    changed = sum(1 for t in out.values() if t != "customer")
    print(f"Verified {len(out)}/{len(df)}: {changed} relabeled away from customer ({changed / max(len(out), 1):.0%})")


def sample(n=40):
    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT author_type, source, LEFT(text_clean, 100) t FROM (
              SELECT *, ROW_NUMBER() OVER (PARTITION BY author_type ORDER BY RANDOM()) rn
              FROM staging.content_enriched WHERE author_type IS NOT NULL) x
            WHERE rn <= %s ORDER BY author_type
            """, conn, params=(max(n // 4, 1),))
    for _, r in df.iterrows():
        print(f"[{r['author_type']:<9}|{r['source']:<15}] {r['t']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()
    if a.sample:
        sample(a.sample)
    elif a.verify:
        verify(a.limit)
    else:
        run(a.limit)


if __name__ == "__main__":
    main()
