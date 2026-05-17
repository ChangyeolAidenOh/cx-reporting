"""Pre-EDA Check 3: Naver DataLab search volume comparison for 4 brands.

Checklist item: Naver DataLab 4사 브랜드 검색량 상대 비교

Usage:
    python scripts/eda_naver_datalab.py
"""

import os
import json

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

HEADERS = {
    "X-Naver-Client-Id": CLIENT_ID,
    "X-Naver-Client-Secret": CLIENT_SECRET,
    "Content-Type": "application/json",
}

# Naver DataLab allows up to 5 keyword groups per request
KEYWORD_GROUPS = [
    {
        "groupName": "메트라이프",
        "keywords": ["메트라이프", "메트라이프생명", "메트라이프보험"],
    },
    {
        "groupName": "삼성생명",
        "keywords": ["삼성생명", "삼성생명보험"],
    },
    {
        "groupName": "한화생명",
        "keywords": ["한화생명", "한화생명보험"],
    },
    {
        "groupName": "교보생명",
        "keywords": ["교보생명", "교보생명보험"],
    },
]


def fetch_datalab_trend(start_date, end_date, time_unit="week"):
    """Fetch Naver DataLab search trend data."""
    url = "https://openapi.naver.com/v1/datalab/search"
    body = {
        "startDate": start_date,
        "endDate": end_date,
        "timeUnit": time_unit,
        "keywordGroups": KEYWORD_GROUPS,
    }

    resp = requests.post(url, headers=HEADERS, json=body, timeout=10)
    resp.raise_for_status()
    return resp.json()


def parse_trend_data(response):
    """Parse DataLab response into DataFrame."""
    results = response.get("results", [])
    all_rows = []

    for group in results:
        brand = group["title"]
        for point in group.get("data", []):
            all_rows.append({
                "brand": brand,
                "period": point["period"],
                "ratio": point["ratio"],
            })

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df["period"] = pd.to_datetime(df["period"])
    return df


def run_datalab_check():
    """Run Naver DataLab search volume comparison."""
    if not CLIENT_ID or not CLIENT_SECRET:
        print("[ERROR] NAVER_CLIENT_ID / NAVER_CLIENT_SECRET not set in .env")
        return

    # Fetch last 12 months of weekly data
    end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    start_date = (pd.Timestamp.now() - pd.DateOffset(months=12)).strftime("%Y-%m-%d")

    print(f"Fetching Naver DataLab trends: {start_date} ~ {end_date}")

    response = fetch_datalab_trend(start_date, end_date, time_unit="week")
    df = parse_trend_data(response)

    if df.empty:
        print("[ERROR] No data returned from Naver DataLab")
        return

    # Summary stats
    print("\n" + "=" * 70)
    print("Pre-EDA Check 3: Naver DataLab Brand Search Volume (relative)")
    print(f"Period: {start_date} ~ {end_date} (weekly)")
    print("=" * 70)

    summary = df.groupby("brand")["ratio"].agg(["mean", "std", "min", "max"])
    summary.columns = ["Mean", "Std", "Min", "Max"]
    summary = summary.sort_values("Mean", ascending=False)

    print(f"\n{'Brand':<12} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'SoV%'}")
    print("-" * 62)

    total_mean = summary["Mean"].sum()
    for brand, row in summary.iterrows():
        sov = (row["Mean"] / total_mean * 100) if total_mean > 0 else 0
        print(
            f"{brand:<12} "
            f"{row['Mean']:<10.1f} "
            f"{row['Std']:<10.1f} "
            f"{row['Min']:<10.1f} "
            f"{row['Max']:<10.1f} "
            f"{sov:.1f}%"
        )

    # Recent trend (last 4 weeks)
    recent = df[df["period"] >= df["period"].max() - pd.Timedelta(weeks=4)]
    if not recent.empty:
        print("\nRecent 4-week trend:")
        pivot = recent.pivot(index="period", columns="brand", values="ratio")
        print(pivot.to_string())

    # Decision
    print("\n" + "=" * 70)
    print("Decision Points:")
    metlife_mean = summary.loc["메트라이프", "Mean"] if "메트라이프" in summary.index else 0
    if metlife_mean < 5:
        print(f"  MetLife mean ratio: {metlife_mean:.1f} (LOW)")
        print("  -> MetLife brand search volume is relatively low vs domestic insurers.")
        print("  -> Expected for foreign brand. Content-to-search signal may be weaker.")
        print("  -> Consider: use absolute Naver DataLab volume (not relative) if available.")
    else:
        print(f"  MetLife mean ratio: {metlife_mean:.1f}")
        print("  -> Sufficient search volume for trend analysis.")

    # Save
    out_path = "data/raw/pre_eda_naver_datalab.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    response_clean = {
        "period": f"{start_date} ~ {end_date}",
        "summary": summary.to_dict(),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(response_clean, f, ensure_ascii=False, indent=2, default=str)

    csv_path = "data/raw/pre_eda_naver_datalab.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"\n  Raw results saved: {out_path}")
    print(f"  CSV saved: {csv_path}")


if __name__ == "__main__":
    run_datalab_check()
