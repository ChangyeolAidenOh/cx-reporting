"""Reassign brand for blog/cafe docs whose title names a different single brand.

Why: blog/cafe docs get their brand from the collection query ("삼성화재 후기"),
but comparison-heavy areas (태아보험 especially) return posts that are about
another company. Sample review found 12% of 삼성화재-assigned docs with only
another brand in the title, and 20/20 of those were genuinely about that
other brand. Brand is an input to the brand x category test, so this is a
fourth confound next to query asymmetry, author type and source mix.

Rule (title only, body mentions every brand in comparison posts):
  - exactly one brand named in the title and it is not the assigned brand
      -> brand := named brand, brand_source = 'title'
  - two or more brands named -> keep (assigned brand is usually among them)
  - no brand named -> keep the query brand
Original value is preserved in brand_query.

Usage:
    python -m scripts.reassign_brand_by_title            dry-run
    python -m scripts.reassign_brand_by_title --apply
"""

import sys
import re
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from config.db import get_conn

BRAND_PATTERNS = {
    "kb": r"(?<![a-z])kb(?![a-z])|케이비손해",
    "samsung_fire": r"삼성 ?화재|삼성 ?다이렉트",
    "db": r"(?<![a-z])db(?![a-z])|디비손해",
    "hyundai": r"현대 ?해상",
    "meritz": r"메리츠",
}
COMPILED = {b: re.compile(p, flags=re.I) for b, p in BRAND_PATTERNS.items()}


def mentioned(title):
    return [b for b, rx in COMPILED.items() if rx.search(title)]


def load():
    with get_conn() as conn:
        df = pd.read_sql("""
            SELECT id, brand, source, author_type, topic_label, COALESCE(title, '') AS title
            FROM staging.content_enriched
            WHERE source IN ('blog', 'cafearticle')
        """, conn)
    df["m"] = df["title"].map(mentioned)
    df["n_m"] = df["m"].map(len)
    df["status"] = "none"
    own = df.apply(lambda r: r["brand"] in r["m"], axis=1)
    df.loc[own & (df["n_m"] > 1), "status"] = "own+other"
    df.loc[own & (df["n_m"] == 1), "status"] = "own_only"
    df.loc[~own & (df["n_m"] == 1), "status"] = "other_only"
    df.loc[~own & (df["n_m"] > 1), "status"] = "others_multi"
    df["new_brand"] = df["brand"]
    change = df["status"] == "other_only"
    df.loc[change, "new_brand"] = df.loc[change, "m"].map(lambda m: m[0])
    return df


def dry_run(df):
    print("Title mention status by assigned brand (blog+cafe, all labels):")
    print(pd.crosstab(df["brand"], df["status"], margins=True).to_string())
    tested = df[(df["topic_label"] != "noise")]
    print("\nSame, non-noise only:")
    print(pd.crosstab(tested["brand"], tested["status"], margins=True).to_string())
    ch = df[df["new_brand"] != df["brand"]]
    print(f"\nWould change: {len(ch)} docs ({len(ch[ch['topic_label'] != 'noise'])} non-noise)")
    print(pd.crosstab(ch["brand"], ch["new_brand"]).to_string())
    print("\nSample (40, non-noise):")
    s = ch[ch["topic_label"] != "noise"].sample(min(40, len(ch)), random_state=11)
    for _, r in s.sort_values("brand").iterrows():
        print(f"[{r['id']}][{r['brand']} -> {r['new_brand']}][{r['author_type']}][{r['topic_label']}]  {r['title'][:70]}")


def apply(df):
    ch = df[df["new_brand"] != df["brand"]]
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE staging.content_enriched ADD COLUMN IF NOT EXISTS brand_query VARCHAR(50)")
            cur.execute("ALTER TABLE staging.content_enriched ADD COLUMN IF NOT EXISTS brand_source VARCHAR(20)")
            cur.execute("UPDATE staging.content_enriched SET brand_query = brand WHERE brand_query IS NULL")
            cur.execute("UPDATE staging.content_enriched SET brand_source = 'query' WHERE brand_source IS NULL")
            for _, r in ch.iterrows():
                cur.execute("""
                    UPDATE staging.content_enriched
                    SET brand = %s, brand_source = 'title', processed_at = NOW()
                    WHERE id = %s""", (r["new_brand"], int(r["id"])))
        conn.commit()
    print(f"Reassigned brand for {len(ch)} docs (brand_source='title', original kept in brand_query)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    df = load()
    if args.apply:
        apply(df)
    else:
        dry_run(df)
