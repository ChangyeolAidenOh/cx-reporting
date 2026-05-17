"""Generate figures for README / portfolio.

Saves 4 key charts to figures/ directory (700x560 canvas).

Usage:
    python scripts/generate_figures.py
"""

import sys
import os
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

FIGURES_DIR = "figures"
WIDTH, HEIGHT = 700, 560

os.makedirs(FIGURES_DIR, exist_ok=True)


def fig_sov_trend():
    """Figure 1: SoV 4-brand trend."""
    df = pd.read_csv("data/exports/brand_search_weekly.csv", parse_dates=["week_start"])

    colors = {
        "삼성생명": "#1428A0",
        "한화생명": "#FF6600",
        "교보생명": "#003DA5",
        "메트라이프": "#00A651",
    }

    fig = go.Figure()
    for brand in ["삼성생명", "한화생명", "교보생명", "메트라이프"]:
        bdf = df[df["brand"] == brand].sort_values("week_start")
        fig.add_trace(go.Scatter(
            x=bdf["week_start"], y=bdf["sov_pct"],
            name=brand, mode="lines",
            line=dict(color=colors.get(brand, "#888"), width=2.5 if brand == "메트라이프" else 1.5),
        ))

    fig.update_layout(
        title="Brand Search Interest — Share of Voice (%)",
        xaxis_title="", yaxis_title="SoV %",
        width=WIDTH, height=HEIGHT,
        legend=dict(orientation="h", y=-0.12),
        template="plotly_white",
        font=dict(size=12),
    )
    path = f"{FIGURES_DIR}/01_sov_trend.png"
    fig.write_image(path, scale=2)
    print(f"Saved: {path}")


def fig_signal_matrix():
    """Figure 2: Engagement vs Search Interest Signal Matrix."""
    df = pd.read_csv("data/exports/content_signal_matrix.csv")

    colors = {"samsung": "#1428A0", "hanwha": "#FF6600", "kyobo": "#003DA5"}

    fig = go.Figure()
    for brand in df["brand"].unique():
        bdf = df[df["brand"] == brand]
        fig.add_trace(go.Scatter(
            x=bdf["avg_engagement"], y=bdf["search_signal"],
            mode="markers+text",
            name=brand,
            text=bdf["topic_label"],
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(
                size=bdf["sample_count"].clip(lower=10) * 0.8,
                color=colors.get(brand, "#888"),
                opacity=0.7,
            ),
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)

    fig.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper",
                       text="High Search Signal", showarrow=False,
                       font=dict(size=10, color="gray"))
    fig.add_annotation(x=0.02, y=0.02, xref="paper", yref="paper",
                       text="Low Search Signal", showarrow=False,
                       font=dict(size=10, color="gray"))

    fig.update_layout(
        title="Content Signal Matrix — Engagement vs Search Interest",
        xaxis_title="Avg Engagement (likes + comments)",
        yaxis_title="Search Interest Signal (WoW %)",
        width=WIDTH, height=HEIGHT,
        legend=dict(orientation="h", y=-0.12),
        template="plotly_white",
        font=dict(size=12),
    )
    path = f"{FIGURES_DIR}/02_signal_matrix.png"
    fig.write_image(path, scale=2)
    print(f"Saved: {path}")


def fig_channel_sentiment():
    """Figure 3: Channel-dependent sentiment comparison."""
    data = {
        "Channel": ["Blog/Cafe VoC", "Blog/Cafe VoC", "Blog/Cafe VoC",
                     "Foreign Insurer VoC", "Foreign Insurer VoC", "Foreign Insurer VoC",
                     "App Store Reviews", "App Store Reviews", "App Store Reviews"],
        "Sentiment": ["Positive", "Neutral", "Negative"] * 3,
        "Ratio": [24.5, 66.8, 8.8,
                  36.6, 44.0, 19.4,
                  19.3, 8.0, 72.7],
    }
    df = pd.DataFrame(data)

    color_map = {"Positive": "#2ecc71", "Neutral": "#95a5a6", "Negative": "#e74c3c"}

    fig = go.Figure()
    for sentiment in ["Positive", "Neutral", "Negative"]:
        sdf = df[df["Sentiment"] == sentiment]
        fig.add_trace(go.Bar(
            x=sdf["Channel"], y=sdf["Ratio"],
            name=sentiment,
            marker_color=color_map[sentiment],
            text=[f"{v:.1f}%" for v in sdf["Ratio"]],
            textposition="inside",
            textfont=dict(size=12, color="white"),
        ))

    fig.update_layout(
        title="Channel-Dependent Perception Gap — MetLife Sentiment by VoC Source",
        xaxis_title="", yaxis_title="Ratio (%)",
        barmode="stack",
        width=WIDTH, height=HEIGHT,
        legend=dict(orientation="h", y=-0.12),
        template="plotly_white",
        font=dict(size=12),
    )
    path = f"{FIGURES_DIR}/03_channel_sentiment.png"
    fig.write_image(path, scale=2)
    print(f"Saved: {path}")


def fig_lead_lag():
    """Figure 4: Lead-Lag cross-correlation for MetLife."""
    lag_path = "data/exports/lead_lag_results.json"
    with open(lag_path, "r") as f:
        lag_data = json.load(f)

    metlife_lags = lag_data.get("metlife", {})
    if not metlife_lags:
        print("No MetLife lead-lag data found")
        return

    lags = sorted(metlife_lags.keys(), key=lambda x: int(x))
    values = [metlife_lags[l] for l in lags]
    lag_ints = [int(l) for l in lags]

    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]
    # Highlight best lag
    best_idx = max(range(len(values)), key=lambda i: abs(values[i]))
    colors[best_idx] = "#f39c12"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=lag_ints, y=values,
        marker_color=colors,
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        textfont=dict(size=9),
    ))

    fig.add_hline(y=0.3, line_dash="dot", line_color="green", opacity=0.4,
                  annotation_text="r=0.3", annotation_position="top right")
    fig.add_hline(y=-0.3, line_dash="dot", line_color="red", opacity=0.4,
                  annotation_text="r=-0.3", annotation_position="bottom right")
    fig.add_hline(y=0, line_color="gray", opacity=0.3)

    fig.add_annotation(
        x=2, y=metlife_lags.get("2", 0),
        text="Best: lag=+2w, r=0.22",
        showarrow=True, arrowhead=2,
        font=dict(size=11, color="#f39c12"),
    )

    fig.update_layout(
        title="MetLife Lead-Lag Cross-Correlation (MSTL Residual)",
        xaxis_title="Lag (weeks) — positive = content leads search",
        yaxis_title="Correlation (r)",
        width=WIDTH, height=HEIGHT,
        template="plotly_white",
        font=dict(size=12),
        yaxis=dict(range=[-0.4, 0.4]),
    )
    path = f"{FIGURES_DIR}/04_lead_lag_metlife.png"
    fig.write_image(path, scale=2)
    print(f"Saved: {path}")


def main():
    print("Generating README figures...\n")
    fig_sov_trend()
    fig_signal_matrix()
    fig_channel_sentiment()
    fig_lead_lag()
    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
