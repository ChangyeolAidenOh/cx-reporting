"""Tab 3: VoC Monitor - 감성 추이, 토픽별 감성, 이상 이벤트."""

import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def load_data():
    sentiment_path = "data/exports/voc_sentiment.csv"
    enriched_path = "data/processed/content_enriched_text.csv"
    sent_results_path = "data/processed/sentiment_results.csv"

    sentiment = pd.read_csv(sentiment_path, parse_dates=["week_start"]) if os.path.exists(sentiment_path) else pd.DataFrame()
    enriched = pd.read_csv(enriched_path) if os.path.exists(enriched_path) else pd.DataFrame()
    sent_results = pd.read_csv(sent_results_path) if os.path.exists(sent_results_path) else pd.DataFrame()

    return sentiment, enriched, sent_results


def render():
    sentiment, enriched, sent_results = load_data()

    if sentiment.empty:
        st.warning("VoC sentiment data not available.")
        return

    # Brand filter
    brands = sorted(sentiment["brand"].unique())
    selected_brand = st.selectbox("Brand Filter", ["All"] + brands, key="voc_brand")

    if selected_brand != "All":
        sentiment = sentiment[sentiment["brand"] == selected_brand]

    st.markdown("---")

    # Sentiment trend
    st.subheader("Sentiment Trend (Weekly)")

    fig_sent = go.Figure()
    if selected_brand == "All":
        avg_sent = sentiment.groupby("week_start")[["positive_ratio", "neutral_ratio", "negative_ratio"]].mean().reset_index()
    else:
        avg_sent = sentiment.copy()

    fig_sent.add_trace(go.Scatter(
        x=avg_sent["week_start"], y=avg_sent["positive_ratio"],
        name="Positive", line=dict(color="#2ecc71"), stackgroup="one",
    ))
    fig_sent.add_trace(go.Scatter(
        x=avg_sent["week_start"], y=avg_sent["neutral_ratio"],
        name="Neutral", line=dict(color="#95a5a6"), stackgroup="one",
    ))
    fig_sent.add_trace(go.Scatter(
        x=avg_sent["week_start"], y=avg_sent["negative_ratio"],
        name="Negative", line=dict(color="#e74c3c"), stackgroup="one",
    ))
    fig_sent.update_layout(
        height=400, yaxis_tickformat=".0%",
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_sent, use_container_width=True)

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    latest = sentiment[sentiment["week_start"] == sentiment["week_start"].max()]

    with col1:
        st.metric("Avg Positive", f"{latest['positive_ratio'].mean():.1%}")
    with col2:
        st.metric("Avg Negative", f"{latest['negative_ratio'].mean():.1%}")
    with col3:
        st.metric("Total Docs (latest)", f"{int(latest['total_docs'].sum())}")
    with col4:
        st.metric("Weeks Tracked", f"{sentiment['week_start'].nunique()}")

    st.markdown("---")

    # Sentiment by brand comparison
    st.subheader("Sentiment by Brand")

    brand_avg = sentiment.groupby("brand")[["positive_ratio", "negative_ratio"]].mean().reset_index()
    brand_avg = brand_avg.melt(id_vars="brand", var_name="sentiment", value_name="ratio")
    brand_avg["sentiment"] = brand_avg["sentiment"].map({
        "positive_ratio": "Positive", "negative_ratio": "Negative"
    })

    fig_brand = px.bar(
        brand_avg, x="brand", y="ratio", color="sentiment",
        barmode="group",
        color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c"},
        labels={"brand": "", "ratio": "Avg Ratio", "sentiment": ""},
    )
    fig_brand.update_layout(height=350, yaxis_tickformat=".0%")
    st.plotly_chart(fig_brand, use_container_width=True)

    st.markdown("---")

    # Topic x Sentiment distribution
    if not sent_results.empty and "label" in sent_results.columns:
        st.subheader("Sentiment by Topic Category")

        # Merge topic labels with sentiment results
        if not enriched.empty and "topic_label" in enriched.columns:
            merged = sent_results.merge(
                enriched[["source_id", "topic_label", "brand"]].rename(columns={"source_id": "id"}),
                left_on="id", right_on="id",
                how="left", suffixes=("", "_enriched"),
            )
            merged = merged[merged["topic_label"].notna() & (merged["topic_label"] != "noise")]

            if not merged.empty:
                topic_sent = merged.groupby(["topic_label", "label"]).size().reset_index(name="count")
                fig_topic = px.bar(
                    topic_sent, x="topic_label", y="count", color="label",
                    color_discrete_map={
                        "positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c"
                    },
                    labels={"topic_label": "Topic", "count": "Count", "label": "Sentiment"},
                    barmode="stack",
                )
                fig_topic.update_layout(height=400)
                st.plotly_chart(fig_topic, use_container_width=True)

    st.markdown("---")

    # Anomaly events
    st.subheader("Anomaly Events")

    anomaly_path = "data/exports/content_signal_matrix.csv"  # placeholder
    # Load from DB or show from anomaly_log
    st.info("Anomaly events are displayed in the Weekly Report tab for the selected week.")
