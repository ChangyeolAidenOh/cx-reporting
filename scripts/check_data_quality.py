"""Data quality check for Phase 1 test collection.

Checks:
  1. Naver DataLab: gaps, nulls, ratio range
  2. YouTube: missing fields, comment-to-video ratio
  3. Naver VoC: postdate coverage, brand tagging, description length

Usage:
    python scripts/check_data_quality.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from config.db import get_conn


def separator(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def check_datalab():
    """Check Naver DataLab data quality."""
    separator("1. Naver DataLab (raw.naver_datalab)")

    with get_conn() as conn:
        df = pd.read_sql("SELECT * FROM raw.naver_datalab ORDER BY brand, period", conn)

    print(f"Total rows: {len(df)}")
    print(f"Brands: {df['brand'].nunique()} -> {sorted(df['brand'].unique())}")
    print(f"Period: {df['period'].min()} ~ {df['period'].max()}")
    print(f"Weeks per brand: {df.groupby('brand').size().to_dict()}")

    # Null check
    nulls = df.isnull().sum()
    print(f"\nNull counts:\n{nulls[nulls > 0].to_string()}" if nulls.any() else "\nNo nulls")

    # Ratio range
    print(f"\nRatio stats:")
    for brand in sorted(df["brand"].unique()):
        bdf = df[df["brand"] == brand]
        print(f"  {brand}: min={bdf['ratio'].min():.1f}  max={bdf['ratio'].max():.1f}  "
              f"mean={bdf['ratio'].mean():.1f}  zeros={len(bdf[bdf['ratio'] == 0])}")

    # Gap detection (missing weeks)
    print(f"\nWeekly gap check:")
    for brand in sorted(df["brand"].unique()):
        bdf = df[df["brand"] == brand].sort_values("period")
        bdf["period"] = pd.to_datetime(bdf["period"])
        diffs = bdf["period"].diff().dropna()
        max_gap = diffs.max()
        gaps_over_8d = len(diffs[diffs > pd.Timedelta(days=8)])
        print(f"  {brand}: max gap={max_gap.days}d  gaps>8d={gaps_over_8d}")


def check_youtube():
    """Check YouTube data quality."""
    separator("2. YouTube Videos (raw.youtube_videos)")

    with get_conn() as conn:
        vdf = pd.read_sql("SELECT * FROM raw.youtube_videos", conn)
        cdf = pd.read_sql("SELECT * FROM raw.youtube_comments", conn)

    print(f"Total videos: {len(vdf)}")
    print(f"Total comments: {len(cdf)}")
    print(f"Brands: {vdf['brand'].value_counts().to_dict()}")

    # Null/empty check
    print(f"\nVideo field quality:")
    print(f"  title empty: {len(vdf[vdf['title'].fillna('').str.len() == 0])}")
    print(f"  description empty: {len(vdf[vdf['description'].fillna('').str.len() == 0])}")
    print(f"  published_at null: {vdf['published_at'].isnull().sum()}")
    print(f"  view_count=0: {len(vdf[vdf['view_count'] == 0])}")

    # Comment-to-video ratio
    print(f"\nComment-to-video ratio by brand:")
    for brand in sorted(vdf["brand"].unique()):
        vid_count = len(vdf[vdf["brand"] == brand])
        vid_ids = vdf[vdf["brand"] == brand]["video_id"].tolist()
        com_count = len(cdf[cdf["video_id"].isin(vid_ids)])
        ratio = com_count / vid_count if vid_count > 0 else 0
        print(f"  {brand}: {vid_count} videos, {com_count} comments, "
              f"avg {ratio:.1f} comments/video")

    # Comment text quality
    separator("2b. YouTube Comments (raw.youtube_comments)")
    print(f"Total comments: {len(cdf)}")
    print(f"  text empty: {len(cdf[cdf['text_original'].fillna('').str.len() == 0])}")
    print(f"  avg length: {cdf['text_original'].fillna('').str.len().mean():.0f} chars")
    print(f"  median length: {cdf['text_original'].fillna('').str.len().median():.0f} chars")
    print(f"  very short (<10 chars): {len(cdf[cdf['text_original'].fillna('').str.len() < 10])}")

    # Date range
    if not vdf.empty:
        print(f"\nVideo date range:")
        vdf["published_at"] = pd.to_datetime(vdf["published_at"])
        for brand in sorted(vdf["brand"].unique()):
            bdf = vdf[vdf["brand"] == brand]
            print(f"  {brand}: {bdf['published_at'].min().date()} ~ "
                  f"{bdf['published_at'].max().date()}")


def check_voc():
    """Check Naver VoC data quality."""
    separator("3. Naver VoC (raw.naver_voc)")

    with get_conn() as conn:
        df = pd.read_sql("SELECT * FROM raw.naver_voc", conn)

    print(f"Total rows: {len(df)}")
    print(f"Sources: {df['source'].value_counts().to_dict()}")
    print(f"Brands: {df['brand'].value_counts().to_dict()}")

    # Postdate coverage
    print(f"\nPostdate coverage:")
    for source in sorted(df["source"].unique()):
        sdf = df[df["source"] == source]
        null_count = sdf["postdate"].isnull().sum()
        coverage = (1 - null_count / len(sdf)) * 100 if len(sdf) > 0 else 0
        print(f"  {source}: {coverage:.1f}% have postdate "
              f"({len(sdf) - null_count}/{len(sdf)})")

    # Date range (where available)
    dated = df[df["postdate"].notna()].copy()
    if not dated.empty:
        dated["postdate"] = pd.to_datetime(dated["postdate"])
        print(f"\nDate range (where available): "
              f"{dated['postdate'].min().date()} ~ {dated['postdate'].max().date()}")

    # Description quality
    print(f"\nDescription field quality:")
    print(f"  empty: {len(df[df['description'].fillna('').str.len() == 0])}")
    print(f"  avg length: {df['description'].fillna('').str.len().mean():.0f} chars")
    print(f"  very short (<20 chars): "
          f"{len(df[df['description'].fillna('').str.len() < 20])}")

    # Title quality
    print(f"\nTitle field quality:")
    print(f"  empty: {len(df[df['title'].fillna('').str.len() == 0])}")
    print(f"  avg length: {df['title'].fillna('').str.len().mean():.0f} chars")

    # Duplicate check
    dup_links = df[df["link"].duplicated(keep=False)]
    print(f"\nDuplicate links: {len(dup_links)} rows ({len(dup_links) // 2} pairs)")

    # Brand tagging accuracy sample
    print(f"\n'unknown' brand sample (first 5):")
    unknowns = df[df["brand"] == "unknown"].head(5)
    for _, row in unknowns.iterrows():
        print(f"  query='{row['query']}' | title='{row['title'][:60]}'")


def main():
    print("Phase 1 Data Quality Check")
    print("=" * 70)

    check_datalab()
    check_youtube()
    check_voc()

    separator("Summary")
    print("Review the above and address any issues before full collection.")
    print("Key decisions:")
    print("  - Cafe postdate missing? -> Handle in staging via link parsing")
    print("  - 'unknown' brands? -> Expand brand patterns or reclassify")
    print("  - Short comments? -> Filter threshold for NLP analysis")


if __name__ == "__main__":
    main()
