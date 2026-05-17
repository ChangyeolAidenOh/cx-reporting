"""Run all pre-EDA checks before main project kickoff.

Checklist:
  [1] Naver Blog/Cafe VoC volume per brand
  [2] YouTube official channel stats
  [3] Naver DataLab search volume comparison

Usage:
    python scripts/run_pre_eda.py
"""

import sys
import importlib


def run_check(module_name, label):
    print(f"\n{'#' * 70}")
    print(f"# {label}")
    print(f"{'#' * 70}")
    try:
        mod = importlib.import_module(module_name)
        func_name = [n for n in dir(mod) if n.startswith("run_")][0]
        getattr(mod, func_name)()
    except Exception as e:
        print(f"[FAILED] {e}")
        return False
    return True


def main():
    sys.path.insert(0, ".")

    checks = [
        ("scripts.eda_naver_blog_volume", "Check 1: Naver Blog/Cafe VoC Volume"),
        ("scripts.eda_youtube_channels", "Check 2: YouTube Channel Stats"),
        ("scripts.eda_naver_datalab", "Check 3: Naver DataLab Search Volume"),
    ]

    results = []
    for module, label in checks:
        ok = run_check(module, label)
        results.append((label, ok))

    print(f"\n{'=' * 70}")
    print("Pre-EDA Summary")
    print(f"{'=' * 70}")
    for label, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}")

    print("\nNext step: Review results in data/raw/pre_eda_*.json")
    print("Then update config/settings.py with confirmed channel IDs and keywords.")


if __name__ == "__main__":
    main()
