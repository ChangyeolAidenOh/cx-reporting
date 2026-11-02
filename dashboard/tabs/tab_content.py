"""Tab 2: Content & Category — Signal matrix, 브랜드×카테고리 검정, lead-lag."""

import os
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from config.settings import BRANDS, TARGET_BRAND

BRAND_KEYS = list(BRANDS.keys())


def load():
    def rd(p, **kw):
        return pd.read_csv(p, **kw) if os.path.exists(p) else pd.DataFrame()
    matrix = rd("data/exports/content_signal_matrix.csv")
    test = rd("data/exports/brand_category_test.csv")
    perf = rd("data/exports/content_performance.csv", parse_dates=["week_start"])
    lag = json.load(open("data/exports/lead_lag_results.json")) if os.path.exists("data/exports/lead_lag_results.json") else {}
    return matrix, test, perf, lag


def render():
    matrix, test, perf, lag = load()

    # --- Brand x Category test heatmap
    st.subheader("브랜드 × 카테고리 VoC 비중 — 경쟁사 대비 인덱스")
    st.caption("카이제곱 + Bonferroni 보정 z-test. 대칭 수집 서브셋 기준 (타겟 브랜드 상품별 쿼리 제외). "
               "★ = 통계 검증 ✓")
    if not test.empty:
        pv = test.pivot(index="brand", columns="topic_label", values="ratio").reindex(BRAND_KEYS)
        sig = test.pivot(index="brand", columns="topic_label", values="significant").reindex(BRAND_KEYS)
        txt = pv.round(2).astype(str) + np.where(sig.fillna(False), " ★", "")
        fig = go.Figure(go.Heatmap(
            z=np.log2(pv.values.astype(float)), x=pv.columns, y=pv.index,
            colorscale="RdBu_r", zmid=0, text=txt.values, texttemplate="%{text}",
            colorbar=dict(title="log2(ratio)")))
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
        tgt = test[(test["brand"] == TARGET_BRAND) & test["significant"]].sort_values("ratio", ascending=False)
        if not tgt.empty:
            lines = [f"**{r['topic_label']}** {r['ratio']:.2f}x" for _, r in tgt.iterrows()]
            st.info(f"{BRANDS[TARGET_BRAND]['name_kr']} 유의 인덱싱: " + " · ".join(lines))
    else:
        st.info("brand_category_test.csv not available.")

    st.markdown("---")

    # --- Signal matrix
    st.subheader("Engagement vs Search Interest Signal Matrix")
    st.caption("YouTube 댓글 기반 engagement × 게시 익주 검색 WoW. [cluster 시그널] — 인과 아님")
    if not matrix.empty:
        fig = px.scatter(matrix, x="avg_engagement", y="search_signal", color="brand",
                         size="sample_count", text="topic_label", log_x=True,
                         hover_data=["quadrant", "sample_count"],
                         labels={"avg_engagement": "Avg Engagement (log)", "search_signal": "Search Signal (WoW %)"})
        fig.update_traces(textposition="top center", textfont_size=9)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- Content volume
    if not perf.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Weekly Video Count")
            wv = perf.groupby(["week_start", "brand"])["content_count"].sum().reset_index()
            f = px.bar(wv, x="week_start", y="content_count", color="brand", barmode="group",
                       labels={"week_start": "", "content_count": "Videos"})
            f.update_layout(height=320, legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(f, use_container_width=True)
        with c2:
            st.subheader("Avg Engagement Rate (% of subscribers)")
            be = perf.groupby("brand")["avg_engagement"].mean().reset_index()
            f = px.bar(be, x="brand", y="avg_engagement", color="brand",
                       labels={"brand": "", "avg_engagement": "%"})
            f.update_layout(height=320, showlegend=False)
            st.plotly_chart(f, use_container_width=True)

    st.markdown("---")

    # --- Lead-lag
    st.subheader("Lead-Lag Cross-Correlation")
    st.caption("MSTL 잔차 기반. VoC 볼륨 vs 검색 관심도. [정성 시그널] — 광고 물량 교란 가능성으로 해석 보류")
    if lag:
        b = st.selectbox("Brand", list(lag.keys()), index=list(lag.keys()).index(TARGET_BRAND) if TARGET_BRAND in lag else 0)
        ld = pd.DataFrame([{"lag": int(k), "r": v} for k, v in lag[b].items()]).sort_values("lag")
        f = px.bar(ld, x="lag", y="r", color="r", color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                   labels={"lag": "Lag (weeks, + = VoC leads search)", "r": "Correlation"})
        for y, c in [(0.3, "green"), (-0.3, "red")]:
            f.add_hline(y=y, line_dash="dot", line_color=c, opacity=0.5)
        f.update_layout(height=320, coloraxis_showscale=False)
        st.plotly_chart(f, use_container_width=True)
