"""Generate README figures (4 charts) for the 손해보험 5사 analysis.

Usage:
    python scripts/generate_figures.py
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from config.settings import BRANDS, BRAND_KR_TO_KEY, TARGET_BRAND

FIG = "figures"
W, H = 700, 560
os.makedirs(FIG, exist_ok=True)

TARGET_KR = BRANDS[TARGET_BRAND]["name_kr"]
BRAND_ORDER = [v["name_kr"] for v in BRANDS.values()]
COLORS = {"KB손해보험": "#FFB300", "삼성화재": "#1428A0", "DB손해보험": "#008C3C",
          "현대해상": "#003478", "메리츠화재": "#D52B1E"}
CAT_ORDER = ["자동차보험", "실손·건강보험", "보험금 청구·보상", "상품 설계·고객후기", "브랜드 캠페인", "이벤트·프로모션"]


def fig_sov_trend():
    df = pd.read_csv("data/exports/brand_search_weekly.csv", parse_dates=["week_start"])
    fig = go.Figure()
    for b in BRAND_ORDER:
        d = df[df["brand"] == b].sort_values("week_start")
        fig.add_trace(go.Scatter(x=d["week_start"], y=d["sov_pct"], name=b, mode="lines",
                                 line=dict(color=COLORS.get(b, "#888"), width=3 if b == TARGET_KR else 1.5)))
    fig.update_layout(title="Brand Search Interest — Share of Voice (%)", yaxis_title="SoV %",
                      width=W, height=H, template="plotly_white", font=dict(size=12),
                      legend=dict(orientation="h", y=-0.12))
    fig.write_image(f"{FIG}/01_sov_trend.png", scale=2)
    print("Saved 01_sov_trend.png")


def fig_brand_category():
    t = pd.read_csv("data/exports/brand_category_test.csv")
    key_order = list(BRANDS.keys())
    pv = t.pivot(index="brand", columns="topic_label", values="ratio").reindex(key_order)[CAT_ORDER]
    sig = t.pivot(index="brand", columns="topic_label", values="significant").reindex(key_order)[CAT_ORDER]
    txt = pv.round(2).astype(str) + np.where(sig.fillna(False), "★", "")
    ylab = [BRANDS[k]["name_kr"] for k in key_order]
    fig = go.Figure(go.Heatmap(z=np.log2(pv.values.astype(float)), x=CAT_ORDER, y=ylab,
                               colorscale="RdBu_r", zmid=0, text=txt.values, texttemplate="%{text}",
                               textfont=dict(size=12), colorbar=dict(title="log2 ratio")))
    fig.update_layout(title="Brand × Category VoC Index vs. Other 4 Brands<br>"
                            "<sup>★ = Bonferroni-significant (α=0.05/30) · symmetric subset</sup>",
                      width=W, height=H, template="plotly_white", font=dict(size=12),
                      yaxis=dict(autorange="reversed"))
    fig.write_image(f"{FIG}/02_brand_category_index.png", scale=2)
    print("Saved 02_brand_category_index.png")


def fig_sentiment_by_category():
    from config.db import get_conn
    with get_conn() as conn:
        m = pd.read_sql(
            "SELECT topic_label, sentiment_label AS label FROM staging.content_enriched "
            "WHERE brand = %s AND sentiment_label IS NOT NULL AND topic_label != 'noise'",
            conn, params=(TARGET_BRAND,))
    m = m[m["topic_label"].isin(CAT_ORDER)]
    ct = pd.crosstab(m["topic_label"], m["label"], normalize="index").reindex(CAT_ORDER).fillna(0)
    n = m["topic_label"].value_counts().reindex(CAT_ORDER).fillna(0).astype(int)
    fig = go.Figure()
    for lab, c in [("positive", "#2ecc71"), ("neutral", "#95a5a6"), ("negative", "#e74c3c")]:
        if lab in ct:
            fig.add_trace(go.Bar(x=[f"{cat}<br><sup>n={n[cat]}</sup>" for cat in CAT_ORDER], y=ct[lab] * 100,
                                 name=lab.title(), marker_color=c,
                                 text=[f"{v*100:.0f}%" if v > 0.06 else "" for v in ct[lab]],
                                 textposition="inside", textfont=dict(color="white", size=11)))
    fig.update_layout(title=f"{TARGET_KR} — Sentiment by VoC Category", barmode="stack", yaxis_title="%",
                      width=W, height=H, template="plotly_white", font=dict(size=12),
                      legend=dict(orientation="h", y=-0.2))
    fig.write_image(f"{FIG}/03_sentiment_by_category.png", scale=2)
    print("Saved 03_sentiment_by_category.png")


def fig_anomalies():
    import psycopg2
    from config.db import get_conn
    df = pd.read_csv("data/exports/brand_search_weekly.csv", parse_dates=["week_start"])
    with get_conn() as conn:
        an = pd.read_sql("SELECT brand, detected_date FROM mart.anomaly_log", conn)
    an["detected_date"] = pd.to_datetime(an["detected_date"])
    co = an.groupby("detected_date")["brand"].nunique()
    co_dates = co[co >= 4].index

    fig = go.Figure()
    for b in BRAND_ORDER:
        d = df[df["brand"] == b].sort_values("week_start")
        fig.add_trace(go.Scatter(x=d["week_start"], y=d["search_ratio"], name=b, mode="lines",
                                 line=dict(color=COLORS.get(b, "#888"), width=1.5), opacity=0.8))
    for dt in co_dates:
        x0 = (dt - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        x1 = (dt + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        fig.add_vrect(x0=x0, x1=x1, fillcolor="red", opacity=0.12, line_width=0)
        fig.add_annotation(x=dt.strftime("%Y-%m-%d"), y=1.02, yref="paper", text=dt.strftime("%y-%m-%d"),
                           showarrow=False, font=dict(size=9, color="#c0392b"))
    fig.update_layout(title="3-way Anomaly Detection — Industry-wide Co-movement Weeks<br>"
                            "<sup>red bands = ≥4 of 5 brands flagged the same week (holiday weeks)</sup>",
                      yaxis_title="Search Ratio", width=W, height=H, template="plotly_white",
                      font=dict(size=12), legend=dict(orientation="h", y=-0.12))
    fig.write_image(f"{FIG}/04_anomaly_comovement.png", scale=2)
    print("Saved 04_anomaly_comovement.png")


if __name__ == "__main__":
    fig_sov_trend()
    fig_brand_category()
    fig_sentiment_by_category()
    fig_anomalies()
    print(f"\nAll figures saved to {FIG}/")
