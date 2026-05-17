"""Tab 1: Brand Tracker - 4사 검색 관심도 추세, WoW, SoV."""

import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def load_data():
    csv_path = "data/exports/brand_search_weekly.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=["week_start"])
        return df
    return pd.DataFrame()


def render():
    df = load_data()
    if df.empty:
        st.warning("Brand search data not available.")
        return

    # KPI cards for latest week
    latest_week = df["week_start"].max()
    latest = df[df["week_start"] == latest_week].sort_values("sov_pct", ascending=False)

    st.subheader(f"Weekly KPI — {latest_week.strftime('%Y-%m-%d')}")

    cols = st.columns(4)
    for i, (_, row) in enumerate(latest.iterrows()):
        with cols[i]:
            wow = row["wow_change_pct"]
            delta_str = f"{wow:+.1f}%" if pd.notna(wow) else "N/A"
            st.metric(
                label=row["brand"],
                value=f"SoV {row['sov_pct']:.1f}%",
                delta=delta_str,
            )

    st.markdown("---")

    # Search trend chart
    st.subheader("Search Interest Trend")

    fig_trend = px.line(
        df, x="week_start", y="search_ratio", color="brand",
        labels={"week_start": "", "search_ratio": "Search Ratio", "brand": "Brand"},
    )
    fig_trend.update_layout(height=400, legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig_trend, use_container_width=True)

    # SoV trend
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Share of Voice Trend")
        fig_sov = px.area(
            df, x="week_start", y="sov_pct", color="brand",
            labels={"week_start": "", "sov_pct": "SoV %", "brand": "Brand"},
        )
        fig_sov.update_layout(height=350, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_sov, use_container_width=True)

    with col2:
        st.subheader("Latest SoV Distribution")
        fig_pie = px.pie(
            latest, values="sov_pct", names="brand",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

    # WoW change heatmap
    st.subheader("Week-over-Week Change (%)")
    pivot = df.pivot(index="week_start", columns="brand", values="wow_change_pct")
    recent = pivot.tail(12)

    fig_heat = go.Figure(data=go.Heatmap(
        z=recent.values,
        x=recent.columns,
        y=recent.index.strftime("%m-%d"),
        colorscale="RdYlGn",
        zmid=0,
        text=recent.round(1).values,
        texttemplate="%{text}%",
        textfont={"size": 10},
    ))
    fig_heat.update_layout(height=350, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_heat, use_container_width=True)
