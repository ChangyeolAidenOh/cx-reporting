"""Anomaly detection: 3-way ensemble (Z-score, MSTL residual, Isolation Forest).

Detects anomalous weeks in brand search trends.
Stores results in mart.anomaly_log.

Usage:
    python -m analysis.anomaly_detection
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from config.db import get_conn
from config.settings import ANOMALY


def extract():
    with get_conn() as conn:
        df = pd.read_sql(
            "SELECT brand, week_start, search_ratio, wow_change_pct "
            "FROM mart.brand_search_weekly ORDER BY brand, week_start",
            conn,
        )
    df["week_start"] = pd.to_datetime(df["week_start"])
    print(f"Loaded {len(df)} rows from mart.brand_search_weekly")
    return df


def detect_zscore(series, threshold=2.5):
    mean = series.mean()
    std = series.std()
    if std == 0:
        return pd.Series(0, index=series.index)
    return (series - mean) / std


def detect_mstl_residual(series):
    from statsmodels.tsa.seasonal import MSTL
    try:
        if len(series) >= 14:
            result = MSTL(series, periods=[4, 13]).fit()
            return result.resid
        else:
            return series - series.rolling(4, min_periods=1).mean()
    except Exception:
        return series - series.mean()


def detect_isolation_forest(series):
    from sklearn.ensemble import IsolationForest
    X = series.values.reshape(-1, 1)
    clf = IsolationForest(
        contamination=ANOMALY["isolation_forest_contamination"],
        random_state=42,
    )
    preds = clf.fit_predict(X)
    scores = clf.decision_function(X)
    return preds, scores


def run():
    df = extract()
    threshold = ANOMALY["zscore_threshold"]
    min_agree = ANOMALY["min_methods_agree"]

    all_anomalies = []

    for brand in sorted(df["brand"].unique()):
        bdf = df[df["brand"] == brand].set_index("week_start").sort_index()
        series = bdf["search_ratio"]

        # Method 1: Z-score
        zscores = detect_zscore(series, threshold)

        # Method 2: MSTL residual
        mstl_resid = detect_mstl_residual(series)
        mstl_zscore = detect_zscore(mstl_resid, threshold)

        # Method 3: Isolation Forest
        if_preds, if_scores = detect_isolation_forest(series)

        # Combine
        for i, (date, ratio) in enumerate(series.items()):
            z = abs(float(zscores.iloc[i]))
            mz = abs(float(mstl_zscore.iloc[i]))
            ifs = float(if_scores[i])

            methods = 0
            if z > threshold:
                methods += 1
            if mz > threshold:
                methods += 1
            if if_preds[i] == -1:
                methods += 1

            if methods >= min_agree:
                all_anomalies.append({
                    "brand": brand,
                    "detected_date": date,
                    "metric": "search_ratio",
                    "value": float(ratio),
                    "zscore": round(float(zscores.iloc[i]), 4),
                    "mstl_residual": round(float(mstl_resid.iloc[i]), 4),
                    "if_score": round(ifs, 4),
                    "methods_agreed": methods,
                })

    # Save to DB
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM mart.anomaly_log")
            for a in all_anomalies:
                cur.execute(
                    """
                    INSERT INTO mart.anomaly_log
                        (brand, detected_date, metric, value, zscore,
                         mstl_residual, if_score, methods_agreed)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        a["brand"], a["detected_date"], a["metric"],
                        a["value"], a["zscore"], a["mstl_residual"],
                        a["if_score"], a["methods_agreed"],
                    ),
                )
    print(f"\nDetected {len(all_anomalies)} anomalies (min {min_agree}/3 methods agree)")

    for a in all_anomalies:
        print(f"  {a['brand']} | {a['detected_date'].date()} | "
              f"ratio={a['value']:.1f} | z={a['zscore']:.2f} | "
              f"methods={a['methods_agreed']}/3")


if __name__ == "__main__":
    run()
