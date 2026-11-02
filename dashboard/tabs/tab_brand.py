"""Tab 1: Brand Tracker — 검색 관심도 추세, WoW, SoV."""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from config.settings import BRANDS, TARGET_BRAND

TARGET_KR = BRANDS[TARGET_BRAND]["name_kr"]
BRAND_ORDER = [v["name_kr"] for v in BRANDS.values()]


def load_data():
    p = "data/exports/brand_search_weekly.csv"
    return pd.read_csv(p, parse_dates=["week_start"], encoding="utf-8-sig") if os.path.exists(p) else pd.DataFrame()


def render():
    df = load_data()
    if df.empty:
        st.warning("Brand search data not available.")
        return

    latest_week = df["week_start"].max()
    latest = df[df["week_start"] == latest_week].sort_values("sov_pct", ascending=False)

    st.subheader(f"Weekly KPI — {latest_week:%Y-%m-%d}")
    cols = st.columns(len(latest))
    for col, (_, r) in zip(cols, latest.iterrows()):
        with col:
            delta = f"{r['wow_change_pct']:+.1f}%" if pd.notna(r["wow_change_pct"]) else "N/A"
            label = f"⭐ {r['brand']}" if r["brand"] == TARGET_KR else r["brand"]
            st.metric(label=label, value=f"{r['sov_pct']:.1f}%", delta=delta)

    st.markdown("---")
    st.subheader("Search Interest Trend")
    fig = px.line(df, x="week_start", y="search_ratio", color="brand",
                  category_orders={"brand": BRAND_ORDER},
                  labels={"week_start": "", "search_ratio": "Search Ratio", "brand": "Brand"})
    for tr in fig.data:
        if tr.name == TARGET_KR:
            tr.line.width = 3.5
    fig.update_layout(height=400, legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Share of Voice Trend")
        f2 = px.area(df, x="week_start", y="sov_pct", color="brand",
                     category_orders={"brand": BRAND_ORDER},
                     labels={"week_start": "", "sov_pct": "SoV %", "brand": "Brand"})
        f2.update_layout(height=350, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(f2, use_container_width=True)
    with c2:
        st.subheader("Latest SoV Distribution")
        f3 = px.pie(latest, values="sov_pct", names="brand",
                    color_discrete_sequence=px.colors.qualitative.Set2)
        f3.update_layout(height=350)
        st.plotly_chart(f3, use_container_width=True)

    st.subheader("Week-over-Week Change (%)")
    pivot = df.pivot(index="week_start", columns="brand", values="wow_change_pct")[BRAND_ORDER].tail(12)
    f4 = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index.strftime("%m-%d"),
                              colorscale="RdYlGn", zmid=0, text=pivot.round(1).values,
                              texttemplate="%{text}%", textfont={"size": 10}))
    f4.update_layout(height=350, yaxis=dict(autorange="reversed"))
    st.plotly_chart(f4, use_container_width=True)
