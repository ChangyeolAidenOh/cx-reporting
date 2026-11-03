"""Tab 3: VoC Monitor. 감성 추이, 브랜드와 카테고리별 감성, 대표 VoC."""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from config.settings import BRANDS, TARGET_BRAND
from config.tags import TAG_QUAL

SENT_COLORS = {"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c"}


def load():
    def rd(p, **kw):
        return pd.read_csv(p, encoding="utf-8-sig", **kw) if os.path.exists(p) else pd.DataFrame()
    return (rd("data/exports/voc_sentiment.csv", parse_dates=["week_start"]),
            rd("data/processed/sentiment_results.csv"),
            rd("data/exports/content_enriched.csv"),
            rd("data/exports/topic_representatives.csv"))


def render():
    sentiment, sent_results, enriched, reps = load()
    if sentiment.empty:
        st.warning("VoC sentiment data not available.")
        return

    brands = [b for b in sentiment["brand"].unique() if b != "general"]
    sel = st.selectbox("Brand", ["All"] + sorted(brands), key="voc_brand",
                       index=(["All"] + sorted(brands)).index(TARGET_BRAND) if TARGET_BRAND in brands else 0)
    s = sentiment if sel == "All" else sentiment[sentiment["brand"] == sel]

    st.markdown("---")
    st.subheader("Sentiment Trend (Weekly)")
    agg = s.groupby("week_start")[["positive_ratio", "neutral_ratio", "negative_ratio"]].mean().reset_index()
    fig = go.Figure()
    for col, name in [("positive_ratio", "positive"), ("neutral_ratio", "neutral"), ("negative_ratio", "negative")]:
        fig.add_trace(go.Scatter(x=agg["week_start"], y=agg[col], name=name.title(),
                                 line=dict(color=SENT_COLORS[name]), stackgroup="one"))
    fig.update_layout(height=380, yaxis_tickformat=".0%", legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)

    latest = s[s["week_start"] == s["week_start"].max()]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Positive", f"{latest['positive_ratio'].mean():.1%}")
    c2.metric("Avg Negative", f"{latest['negative_ratio'].mean():.1%}")
    c3.metric("Docs (latest week)", f"{int(latest['total_docs'].sum())}")
    c4.metric("Weeks Tracked", f"{s['week_start'].nunique()}")

    st.markdown("---")
    st.subheader("Sentiment by Brand")
    ba = sentiment[sentiment["brand"] != "general"].groupby("brand")[["positive_ratio", "negative_ratio"]].mean().reset_index()
    ba = ba.melt(id_vars="brand", var_name="sentiment", value_name="ratio")
    ba["sentiment"] = ba["sentiment"].map({"positive_ratio": "positive", "negative_ratio": "negative"})
    f = px.bar(ba, x="brand", y="ratio", color="sentiment", barmode="group",
               color_discrete_map=SENT_COLORS, labels={"brand": "", "ratio": "Avg Ratio"})
    f.update_layout(height=320, yaxis_tickformat=".0%")
    st.plotly_chart(f, use_container_width=True)

    st.markdown("---")
    if not sent_results.empty and not enriched.empty and "topic_label" in enriched.columns:
        st.subheader("Sentiment by Category")
        m = sent_results.merge(enriched[["id", "topic_label", "brand"]], on="id", how="left")
        m = m[m["topic_label"].notna() & (m["topic_label"] != "noise")]
        if sel != "All":
            m = m[m["brand"] == sel]
        if not m.empty:
            ts = m.groupby(["topic_label", "label"]).size().reset_index(name="count")
            f = px.bar(ts, x="topic_label", y="count", color="label", barmode="stack",
                       color_discrete_map=SENT_COLORS, labels={"topic_label": "", "count": "Docs", "label": ""})
            f.update_layout(height=360)
            st.plotly_chart(f, use_container_width=True)

    st.markdown("---")
    st.subheader("Representative VoC")
    st.caption(f"브랜드 x 카테고리 x 감성별 대표 문서. 소스 우선순위: YouTube 댓글 -> 카페 -> 블로그. 광고 마커 제외. [{TAG_QUAL}]")
    if not reps.empty:
        r = reps if sel == "All" else reps[reps["brand"] == sel]
        cat = st.selectbox("Category", ["All"] + sorted(r["topic_label"].unique()), key="rep_cat")
        if cat != "All":
            r = r[r["topic_label"] == cat]
        for lab in ["negative", "positive"]:
            sub = r[r["sentiment_label"] == lab].head(5)
            if sub.empty:
                continue
            st.markdown(f"**{lab.title()}**")
            for _, row in sub.iterrows():
                st.markdown(f"- `{row['brand']}` · {row['topic_label']} · {row['source']}  \n  {row['text_clean'][:200]}")
