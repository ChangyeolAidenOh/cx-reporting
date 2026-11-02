"""Weekly pipeline orchestrator.

Usage:
    python -m orchestration.weekly_run                 # full weekly run
    python -m orchestration.weekly_run --skip-collect  # re-process without hitting APIs
    python -m orchestration.weekly_run --no-llm        # dictionary-only sentiment, no report narrative
    python -m orchestration.weekly_run --dry-run       # collect + classify + sentiment, no marts/snapshot/report

Each step is idempotent; a failed step is logged and the run continues where
that is safe (collection), or stops where downstream would be wrong (classify).
"""

import sys
import os
import json
import time
import argparse
import traceback
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from config.db import get_conn

LOG_DIR = "data/exports/runs"
os.makedirs(LOG_DIR, exist_ok=True)


def step(name, fn, *a, stop_on_fail=False, **kw):
    t0 = time.time()
    try:
        out = fn(*a, **kw)
        print(f"  ✓ {name} ({time.time() - t0:.0f}s)")
        return out, None
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        traceback.print_exc()
        if stop_on_fail:
            raise
        return None, str(e)


def latest_week_start(settle_days=2):
    """Latest week whose end (start+6) is at least settle_days before today."""
    from datetime import date, timedelta
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(week_start) FROM mart.brand_search_weekly "
                        "WHERE week_start + 6 <= %s", (date.today() - timedelta(days=settle_days),))
            return cur.fetchone()[0]


def collect():
    from collectors import naver_datalab, youtube, naver_voc
    naver_datalab.collect()                       # full range refetch, upsert
    youtube.collect(max_videos=20)                # recent videos only; comments ON CONFLICT DO NOTHING
    naver_voc.collect(max_per_query=100)          # link UNIQUE -> duplicates ignored


def rebuild_marts():
    from etl.mart import brand_search_weekly, content_performance, voc_sentiment
    from analysis import signal_matrix, anomaly_detection
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM mart.content_performance")
    brand_search_weekly.run()
    content_performance.run()
    voc_sentiment.run()
    signal_matrix.run()
    anomaly_detection.run()


def export_staging():
    import pandas as pd
    with get_conn() as conn:
        pd.read_sql("SELECT id, brand, source, topic_id, topic_label, sentiment_label, sentiment_score, "
                    "label_source, label_conf FROM staging.content_enriched", conn
                    ).to_csv("data/exports/content_enriched.csv", index=False, encoding="utf-8-sig")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-collect", action="store_true")
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    from orchestration import incremental, snapshot
    run_log = {"started": datetime.now().isoformat(), "steps": {}}
    print("=" * 60 + f"\nWeekly run — {run_log['started']}\n" + "=" * 60)

    if not a.skip_collect:
        _, err = step("collect", collect)
        run_log["steps"]["collect"] = err or "ok"

    n_new, _ = step("stage_new", incremental.stage_new, stop_on_fail=True)
    cls, _ = step("classify_new", incremental.classify_new, stop_on_fail=True)
    sent, _ = step("sentiment_new", incremental.sentiment_new, use_llm=not a.no_llm)

    pipeline_stats = {
        "new_docs": n_new or 0,
        "review_rate": (cls or {}).get("review_rate", 0.0),
        "confirmed": (cls or {}).get("confirmed", 0),
        "llm_share": round((sent or {}).get("methods", {}).get("llm", 0) / max((sent or {}).get("n", 1), 1), 4),
    }
    run_log["steps"].update({"new_docs": n_new, "classify": cls, "sentiment": sent})

    if a.dry_run:
        print("\n[dry-run] stopping before marts/snapshot/report")
        print(json.dumps(pipeline_stats, ensure_ascii=False, indent=2))
        return

    _, err = step("rebuild_marts", rebuild_marts, stop_on_fail=True)
    week_start = latest_week_start()
    (snap, diff), _ = step("snapshot", snapshot.run, week_start, pipeline_stats, stop_on_fail=True)
    _, _ = step("export_staging", export_staging)

    if not a.no_llm:
        from report import weekly_report
        _, err = step("report", weekly_report.run)
        run_log["steps"]["report"] = err or "ok"

    run_log["finished"] = datetime.now().isoformat()
    run_log["week_label"] = snap["week_label"]
    run_log["alerts"] = diff.get("alerts", [])
    json.dump(run_log, open(f"{LOG_DIR}/{snap['week_label']}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)

    print("\n" + "=" * 60)
    print(f"Done — {snap['week_label']} | new docs {pipeline_stats['new_docs']} | "
          f"review {pipeline_stats['review_rate']:.0%}")
    if diff.get("alerts"):
        print("Alerts (human review needed):")
        for al in diff["alerts"]:
            print(f"  ! {al}")
    else:
        print("No alerts.")


if __name__ == "__main__":
    main()
