"""Pre-EDA: data volume verification for 손해보험 5사.

Checks:
  [1] Naver Blog/Cafe VoC volume per brand
  [2] YouTube official channel discovery + stats
  [3] Naver DataLab search volume comparison

Usage:
    python scripts/run_pre_eda.py
"""

import sys
import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
import pandas as pd
from dotenv import load_dotenv
from config.settings import BRANDS, TARGET_BRAND

load_dotenv()

CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

NAVER_HEADERS = {
    "X-Naver-Client-Id": CLIENT_ID,
    "X-Naver-Client-Secret": CLIENT_SECRET,
}


def sep(title):
    print(f"\n{'#' * 70}\n# {title}\n{'#' * 70}")


# ============================================================
# Check 1: Blog/Cafe volume
# ============================================================

def check_blog_volume():
    sep("Check 1: Naver Blog/Cafe VoC Volume")
    results = {}
    cutoff = datetime.now() - timedelta(days=30)

    for key, info in BRANDS.items():
        results[key] = {}
        for source in ["blog", "cafearticle"]:
            monthly, total = 0, 0
            for kw in info["naver_keywords"][:2]:
                url = f"https://openapi.naver.com/v1/search/{source}.json"
                resp = requests.get(url, headers=NAVER_HEADERS,
                                    params={"query": f"{kw} 후기", "display": 100, "sort": "date"},
                                    timeout=10)
                data = resp.json()
                total += data.get("total", 0)
                for item in data.get("items", []):
                    pd_str = item.get("postdate", "")
                    if pd_str:
                        try:
                            if datetime.strptime(pd_str, "%Y%m%d") >= cutoff:
                                monthly += 1
                        except ValueError:
                            pass
                time.sleep(0.1)
            results[key][source] = {"monthly_estimate": monthly, "total_available": total}

    print(f"{'Brand':<14} {'Source':<13} {'Monthly':<9} {'Total':<10} Status")
    print("-" * 60)
    for key, srcs in results.items():
        for source, c in srcs.items():
            status = "OK" if c["monthly_estimate"] >= 50 or source == "cafearticle" else "LOW"
            print(f"{key:<14} {source:<13} {c['monthly_estimate']:<9} {c['total_available']:<10} {status}")

    os.makedirs("data/raw", exist_ok=True)
    with open("data/raw/pre_eda_blog_volume.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


# ============================================================
# Check 2: YouTube channels
# ============================================================

def check_youtube():
    sep("Check 2: YouTube Channel Discovery")
    from googleapiclient.discovery import build
    yt = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    results = {}

    for key, info in BRANDS.items():
        print(f"\n{info['name_en']} ({info['name_kr']})")
        resp = yt.search().list(part="snippet", q=f"{info['name_kr']} 공식",
                                type="channel", maxResults=5).execute()
        cands = []
        for item in resp.get("items", []):
            cid = item["snippet"]["channelId"]
            st = yt.channels().list(part="statistics", id=cid).execute()
            subs = int(st["items"][0]["statistics"].get("subscriberCount", 0)) if st.get("items") else 0
            vids = int(st["items"][0]["statistics"].get("videoCount", 0)) if st.get("items") else 0
            cands.append({"channel_id": cid, "title": item["snippet"]["title"],
                          "subscribers": subs, "videos": vids})
            time.sleep(0.2)
        cands.sort(key=lambda c: -c["subscribers"])
        for i, c in enumerate(cands):
            print(f"  [{i}] {c['title']:<30} {c['channel_id']}  subs={c['subscribers']:,}  videos={c['videos']}")
        results[key] = cands

    with open("data/raw/pre_eda_youtube_channels.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n-> Pick the official channel per brand and fill youtube_channel_id in config/settings.py")
    return results


# ============================================================
# Check 3: DataLab
# ============================================================

def check_datalab():
    sep("Check 3: Naver DataLab Search Volume")
    end = pd.Timestamp.now().strftime("%Y-%m-%d")
    start = (pd.Timestamp.now() - pd.DateOffset(months=12)).strftime("%Y-%m-%d")

    groups = [{"groupName": info["name_kr"], "keywords": info["naver_keywords"]}
              for info in BRANDS.values()]
    resp = requests.post("https://openapi.naver.com/v1/datalab/search",
                         headers={**NAVER_HEADERS, "Content-Type": "application/json"},
                         json={"startDate": start, "endDate": end, "timeUnit": "week",
                               "keywordGroups": groups}, timeout=10)
    resp.raise_for_status()

    rows = []
    for g in resp.json().get("results", []):
        for p in g.get("data", []):
            rows.append({"brand": g["title"], "period": p["period"], "ratio": p["ratio"]})
    df = pd.DataFrame(rows)

    summary = df.groupby("brand")["ratio"].agg(["mean", "std", "min", "max"]).sort_values("mean", ascending=False)
    total = summary["mean"].sum()
    print(f"\n{'Brand':<12} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8} SoV%")
    print("-" * 56)
    for b, r in summary.iterrows():
        tag = " <- TARGET" if b == BRANDS[TARGET_BRAND]["name_kr"] else ""
        print(f"{b:<12} {r['mean']:<8.1f} {r['std']:<8.1f} {r['min']:<8.1f} {r['max']:<8.1f} {r['mean']/total*100:.1f}%{tag}")

    df.to_csv("data/raw/pre_eda_naver_datalab.csv", index=False, encoding="utf-8-sig")
    return summary


def main():
    check_blog_volume()
    check_youtube()
    check_datalab()
    print("\nPre-EDA complete. Next: fill youtube_channel_id in config/settings.py")


if __name__ == "__main__":
    main()
