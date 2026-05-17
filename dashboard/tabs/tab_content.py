"""Tab 2: Content Performance - 토픽별 Engagement vs 검색 관심도 매트릭스."""

import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def load_data():
    matrix_path = "data/exports/content_signal_matrix.csv"
    perf_path = "data/exports/content_performance.csv"
    lag_path = "data/exports/lead_lag_results.json"

    matrix = pd.read_csv(matrix_path) if os.path.exists(matrix_path) else pd.DataFrame()
    perf = pd.read_csv(perf_path, parse_dates=["week_start"]) if os.path.exists(perf_path) else pd.DataFrame()

    lag = {}
    if os.path.exists(lag_path):
        with open(lag_path, "r") as f:
            lag = json.load(f)

    return matrix, perf, lag


def render():
    matrix, perf, lag = load_data()

    if matrix.empty:
        st.warning("Signal matrix data not available.")
        return

    # Signal Matrix scatter
    st.subheader("Engagement vs Search Interest Signal Matrix")
    st.caption("Quadrant: content type별 engagement와 검색 관심도 변화 패턴")

    fig = px.scatter(
        matrix, x="avg_engagement", y="search_signal",
        color="brand", size="sample_count",
        hover_data=["topic_label", "quadrant"],
        text="topic_label",
        labels={
            "avg_engagement": "Avg Engagement (likes + comments)",
            "search_signal": "Search Interest Signal (WoW %)",
            "brand": "Brand",
        },
    )
    fig.update_traces(textposition="top center", textfont_size=9)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Matrix Interpretation"):
        st.markdown("""
        - **상단 (Search Signal > 0)**: 콘텐츠 게시 후 검색 관심도 상승 패턴 관찰
        - **하단 (Search Signal < 0)**: 콘텐츠 게시 후 검색 관심도 하락 또는 무관
        - **우측 (High Engagement)**: 높은 참여 반응
        - ⚠️ 이 매트릭스는 인과 관계를 확정하지 않습니다. 패턴 탐색 목적입니다.
        """)

    st.markdown("---")

    # Content performance by brand
    if not perf.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Weekly Content Volume by Brand")
            weekly_vol = perf.groupby(["week_start", "brand"])["content_count"].sum().reset_index()
            fig_vol = px.bar(
                weekly_vol, x="week_start", y="content_count", color="brand",
                labels={"week_start": "", "content_count": "Videos", "brand": "Brand"},
                barmode="group",
            )
            fig_vol.update_layout(height=350, legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig_vol, use_container_width=True)

        with col2:
            st.subheader("Avg Engagement Rate by Brand")
            brand_eng = perf.groupby("brand")["avg_engagement"].mean().reset_index()
            fig_eng = px.bar(
                brand_eng, x="brand", y="avg_engagement",
                labels={"brand": "", "avg_engagement": "Avg Engagement %"},
                color="brand",
            )
            fig_eng.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_eng, use_container_width=True)

    st.markdown("---")

    # Lead-Lag results
    st.subheader("Lead-Lag Cross-Correlation")
    st.caption("MSTL 잔차 기반 시차상관 — 콘텐츠 볼륨 vs 검색 관심도")

    if lag:
        brand_select = st.selectbox("Brand", list(lag.keys()))
        if brand_select:
            lag_data = lag[brand_select]
            lag_df = pd.DataFrame([
                {"lag": int(k), "correlation": v} for k, v in lag_data.items()
            ]).sort_values("lag")

            fig_lag = px.bar(
                lag_df, x="lag", y="correlation",
                labels={"lag": "Lag (weeks)", "correlation": "Correlation"},
                color="correlation",
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
            )
            fig_lag.add_hline(y=0.3, line_dash="dot", line_color="green", opacity=0.5,
                             annotation_text="r=0.3")
            fig_lag.add_hline(y=-0.3, line_dash="dot", line_color="red", opacity=0.5,
                             annotation_text="r=-0.3")
            fig_lag.update_layout(height=350)
            st.plotly_chart(fig_lag, use_container_width=True)

            best_lag = max(lag_data, key=lambda k: abs(lag_data[k]))
            best_r = lag_data[best_lag]
            st.info(
                f"Strongest signal: lag={best_lag} weeks, r={best_r:.4f} "
                f"({'content leads search' if int(best_lag) > 0 else 'search leads content'})"
            )
    else:
        st.info("Lead-lag results not yet available.")
