"""Weekly pipeline orchestrator.

Usage:
    python -m orchestration.weekly_run                 # full weekly run
    python -m orchestration.weekly_run --skip-collect  # re-process without hitting APIs
    python -m orchestration.weekly_run --no-llm        # no Haiku/Sonnet steps, no report narrative
    python -m orchestration.weekly_run --dry-run       # collect + stage + classify + author steps; no sentiment/marts/snapshot/report

Step order:
    collect -> stage_new -> resolve_brands_new -> classify_new (category)
    -> classify_author_new -> llm_author_review -> apply_author_filter
    -> llm_tiebreak (category review docs) -> sentiment_new
    -> rebuild_marts -> snapshot -> export -> report

apply_author_filter runs before llm_tiebreak so docs by unrelated authors are
never sent to the category tie-break. With --no-llm, docs below the author gate
stay author_type NULL (author_type_src='author_review') and are therefore
excluded from the customer-only tests until a run with LLM labels them.

Target week = latest *settled* week: Naver DataLab revises the most recent
week for a few days, so a week is used only when its end is >= SETTLE_DAYS ago.
Run mid-week (Wed) so Mon/Tue revisions are already in.

Each step is idempotent; a failed step is logged and the run continues where
that is safe (collection), or stops where downstream would be wrong (classify).
"""

import sys
import os
import json
import time
import argparse
import traceback
from datetime import datetime, date, timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from config.db import get_conn

LOG_DIR = "data/exports/runs"
SETTLE_DAYS = 3
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


def latest_settled_week(settle_days=SETTLE_DAYS):
    """Latest week whose end (start+6) is at least settle_days before today."""
    cutoff = date.today() - timedelta(days=settle_days)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(week_start) FROM mart.brand_search_weekly WHERE week_start + 6 <= %s",
                        (cutoff,))
            return cur.fetchone()[0]


def week_label_of(d):
    return f"{d.year}-W{d.isocalendar()[1]:02d}"


def collect():
    from collectors import naver_datalab, youtube, naver_voc
    naver_datalab.collect()                       # full range refetch; upsert + append to history
    youtube.collect(max_videos=20)                # recent videos; comments ON CONFLICT DO NOTHING
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
        pd.read_sql("SELECT id, brand, brand_source, source, topic_id, topic_label, sentiment_label, "
                    "sentiment_score, label_source, label_conf, author_type, author_type_src "
                    "FROM staging.content_enriched", conn
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

    # 1. collect
    if not a.skip_collect:
        _, err = step("collect", collect)
        run_log["steps"]["collect"] = err or "ok"

    # 2. stage / brand rule / category / author / filters / sentiment
    n_new, _ = step("stage_new", incremental.stage_new, stop_on_fail=True)
    br, _ = step("resolve_brands_new", incremental.resolve_brands_new)
    br = br or {}
    cls, _ = step("classify_new", incremental.classify_new, stop_on_fail=True)
    cls = cls or {}
    au, _ = step("classify_author_new", incremental.classify_author_new, stop_on_fail=True)
    au = au or {}
    aur = {}
    if not a.no_llm:
        aur, _ = step("llm_author_review", incremental.llm_author_review)
        aur = aur or {}
    # safety stop: the author filter may only touch docs staged this run (plus a small margin)
    af, _ = step("apply_author_filter", incremental.apply_author_filter,
                 max_affected=max((n_new or 0), 50))
    af = af or {}
    tb = {}
    if not a.no_llm:
        tb, _ = step("llm_tiebreak", incremental.llm_tiebreak)
        tb = tb or {}
    sent = {}
    if a.dry_run:
        print("  - sentiment_new skipped (dry-run writes no sentiment)")
    else:
        sent, _ = step("sentiment_new", incremental.sentiment_new, use_llm=not a.no_llm)
        sent = sent or {}

    pipeline_stats = {
        "new_docs": n_new or 0,
        "brand_reassigned": br.get("reassigned", 0),
        "confirmed": cls.get("confirmed", 0),
        "review_rate": cls.get("review_rate", 0.0),              # classifier-only, before tie-break
        "rescued_by_llm": tb.get("rescued", 0),
        "final_review": max(cls.get("review", 0) - tb.get("rescued", 0), 0),
        "author_web_docs": au.get("n", 0),
        "author_confirmed": au.get("confirmed", 0),
        "author_review_rate": round(au.get("review", 0) / max(au.get("n", 1), 1), 4),
        "author_llm_labeled": aur.get("labeled", 0),
        "author_filtered": af.get("n", 0),
        "llm_share": round(sent.get("methods", {}).get("llm", 0) / max(sent.get("n", 1), 1), 4),
    }
    run_log["steps"].update({"new_docs": n_new, "brands": br, "classify": cls, "author": au,
                             "author_review": aur, "author_filter": af, "tiebreak": tb, "sentiment": sent})

    if a.dry_run:
        print("\n[dry-run] stopping before marts/snapshot/report")
        print(json.dumps(pipeline_stats, ensure_ascii=False, indent=2))
        return

    # 3. marts
    step("rebuild_marts", rebuild_marts, stop_on_fail=True)

    # 4. snapshot on the latest settled week
    week_start = latest_settled_week()
    if week_start is None:
        raise RuntimeError("no settled week available")
    week_label = week_label_of(week_start)
    print(f"  target week: {week_label} ({week_start}, settle={SETTLE_DAYS}d)")
    (snap, diff), _ = step("snapshot", snapshot.run, week_start, pipeline_stats, stop_on_fail=True)
    step("export_staging", export_staging)

    # 5. report on the same week
    if not a.no_llm:
        from report import weekly_report
        _, err = step("report", weekly_report.run, week=week_label)
        run_log["steps"]["report"] = err or "ok"

    run_log.update({"finished": datetime.now().isoformat(), "week_label": week_label,
                    "alerts": diff.get("alerts", [])})
    json.dump(run_log, open(f"{LOG_DIR}/{week_label}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)

    print("\n" + "=" * 60)
    print(f"Done — {week_label} | new {pipeline_stats['new_docs']} | "
          f"category review {pipeline_stats['review_rate']:.0%} -> rescued {pipeline_stats['rescued_by_llm']} "
          f"-> final review {pipeline_stats['final_review']} | "
          f"author review {pipeline_stats['author_review_rate']:.0%}, filtered {pipeline_stats['author_filtered']}")
    if diff.get("alerts"):
        print("Alerts (human review needed):")
        for al in diff["alerts"]:
            print(f"  ! {al}")
    else:
        print("No alerts.")


if __name__ == "__main__":
    main()
