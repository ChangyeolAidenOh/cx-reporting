"""ETL: raw.naver_datalab -> mart.brand_search_weekly

Computes:
  - Week-over-Week (WoW) change %
  - Share of Voice (SoV) % per week

Usage:
    python -m etl.mart.brand_search_weekly
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from config.db import get_conn


def extract():
    """Load raw DataLab data."""
    with get_conn() as conn:
        df = pd.read_sql(
            "SELECT brand, period, ratio FROM raw.naver_datalab ORDER BY brand, period",
            conn,
        )
    df["period"] = pd.to_datetime(df["period"])
    print(f"Extracted {len(df)} rows from raw.naver_datalab")
    return df


def transform(df):
    """Compute WoW change and SoV."""
    # WoW change per brand
    df = df.sort_values(["brand", "period"])
    df["prev_ratio"] = df.groupby("brand")["ratio"].shift(1)
    df["wow_change_pct"] = (
        (df["ratio"] - df["prev_ratio"]) / df["prev_ratio"] * 100
    ).round(4)

    # SoV per week
    week_totals = df.groupby("period")["ratio"].sum().rename("week_total")
    df = df.merge(week_totals, on="period", how="left")
    df["sov_pct"] = (df["ratio"] / df["week_total"] * 100).round(2)

    # Clean up
    result = df[["brand", "period", "ratio", "wow_change_pct", "sov_pct"]].copy()
    result.rename(columns={"period": "week_start", "ratio": "search_ratio"}, inplace=True)

    # First week has no WoW
    print(f"Transformed {len(result)} rows ({result['wow_change_pct'].isna().sum()} null WoW for first week)")
    return result


def load(df):
    """Upsert into mart.brand_search_weekly."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            for _, row in df.iterrows():
                cur.execute(
                    """
                    INSERT INTO mart.brand_search_weekly
                        (brand, week_start, search_ratio, wow_change_pct, sov_pct)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (brand, week_start)
                    DO UPDATE SET search_ratio = EXCLUDED.search_ratio,
                                  wow_change_pct = EXCLUDED.wow_change_pct,
                                  sov_pct = EXCLUDED.sov_pct
                    """,
                    (
                        row["brand"],
                        row["week_start"].date(),
                        float(row["search_ratio"]),
                        None if pd.isna(row["wow_change_pct"]) else float(row["wow_change_pct"]),
                        float(row["sov_pct"]),
                    ),
                )
    print(f"Loaded {len(df)} rows to mart.brand_search_weekly")


def save_csv(df, path="data/exports/brand_search_weekly.csv"):
    """CSV export for Streamlit Cloud fallback."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"CSV exported: {path}")


def run():
    df = extract()
    result = transform(df)
    load(result)
    save_csv(result)

    # Summary
    print("\nSoV summary (latest week):")
    latest = result[result["week_start"] == result["week_start"].max()]
    for _, row in latest.sort_values("sov_pct", ascending=False).iterrows():
        wow = f"{row['wow_change_pct']:+.1f}%" if pd.notna(row["wow_change_pct"]) else "N/A"
        print(f"  {row['brand']}: SoV {row['sov_pct']:.1f}%  WoW {wow}")


if __name__ == "__main__":
    run()
