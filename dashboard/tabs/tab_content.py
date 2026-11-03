"""Tab 2: Content & Category. 브랜드 x 카테고리 검정(세그먼트별), signal matrix, lead-lag."""

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
from config.tags import TAG_STAT, TAG_CLUSTER, TAG_QUAL

BRAND_KEYS = list(BRANDS.keys())
CAT_ORDER = ["자동차보험", "실손·건강보험", "보험금 청구·보상", "상품 설계·고객후기", "태아·어린이보험",
             "브랜드 캠페인", "이벤트·프로모션"]
SEGMENTS = {
    "web_customer": "블로그·카페 · 고객 작성 글 (상품·여정 카테고리의 기준)",
    "web_advisor": "블로그·카페 · 설계사·비교 콘텐츠 (공급자 측 신호)",
    "youtube": "YouTube 댓글 (캠페인·이벤트 반응의 기준)",
    "all_pooled": "전체 풀링 (참고용, 채널 규모가 지배하므로 발견에 인용하지 않음)",
}
SMALL_N = 30


def load():
    def rd(p, **kw):
        return pd.read_csv(p, encoding="utf-8-sig", **kw) if os.path.exists(p) else pd.DataFrame()
    matrix = rd("data/exports/content_signal_matrix.csv")
    test = rd("data/exports/brand_category_test.csv")
    perf = rd("data/exports/content_performance.csv", parse_dates=["week_start"])
    lag = json.load(open("data/exports/lead_lag_results.json")) if os.path.exists("data/exports/lead_lag_results.json") else {}
    return matrix, test, perf, lag


def _bool(s):
    return s.astype(str).str.lower().eq("true")


def _prep(test):
    t = test.copy()
    for c in ["note", "loo_fragile_without"]:
        if c in t.columns:
            t[c] = t[c].fillna("")
    for c in ["significant", "robust"]:
        if c in t.columns:
            t[c] = _bool(t[c])
        else:
            t[c] = False
    if "segment" not in t.columns:
        t["segment"] = "all_pooled"
    return t


def render_category_test(test):
    st.subheader("브랜드 x 카테고리 VoC 비중: 경쟁사 대비 인덱스")
    if test.empty or "ratio" not in test.columns:
        st.info("brand_category_test.csv not available.")
        return
    t = _prep(test)
    segs = [s for s in SEGMENTS if s in t["segment"].unique()]
    seg = st.selectbox("세그먼트", segs, index=segs.index("web_customer") if "web_customer" in segs else 0,
                       format_func=lambda s: f"{s}: {SEGMENTS[s]}", key="cat_seg")
    s = t[t["segment"] == seg]
    cats = [c for c in CAT_ORDER if c in s["topic_label"].unique()]
    piv = lambda col: s.pivot(index="brand", columns="topic_label", values=col).reindex(index=BRAND_KEYS, columns=cats)

    ratio, robust, sig = piv("ratio"), piv("robust").fillna(False), piv("significant").fillna(False)
    n_cell = piv("n_brand_cat")
    shrunk = piv("ratio_shrunk") if "ratio_shrunk" in s.columns else ratio
    note = piv("note").fillna("")

    shown = ratio.copy()
    small = n_cell < SMALL_N
    shown = shown.where(~small, shrunk)
    txt = shown.round(2).astype(str)
    txt = txt + np.where(robust, " 패스", np.where(sig, " 취약", np.where(small, " 소표본", "")))
    txt = txt.where(note != "n<5", "미검정")
    z = np.log2(shown.values.astype(float))
    z = np.where(note.values == "n<5", np.nan, z)

    hover = []
    for b in BRAND_KEYS:
        row = []
        for c in cats:
            r = s[(s["brand"] == b) & (s["topic_label"] == c)]
            if r.empty:
                row.append("")
                continue
            r = r.iloc[0]
            parts = [f"{b} · {c}", f"n={int(r['n_brand_cat'])} / {int(r['n_brand'])}",
                     f"share {r['share_brand']:.1%} vs others {r['share_others']:.1%}"]
            if np.isfinite(r.get("ratio", np.nan)):
                parts.append(f"ratio {r['ratio']:.2f}x")
            if int(r["n_brand_cat"]) < SMALL_N and np.isfinite(r.get("ratio_shrunk", np.nan)):
                parts.append(f"shrunk {r['ratio_shrunk']:.2f}x [{r['ratio_ci_low']:.2f}, {r['ratio_ci_high']:.2f}]")
            if r["robust"]:
                parts.append(f"robust (LOO {r['ratio_loo_min']:.2f}-{r['ratio_loo_max']:.2f}x)")
            elif r["significant"]:
                parts.append(f"significant but fragile: drops without {r['loo_fragile_without']}")
            if np.isfinite(r.get("mde_ratio_over", np.nan)):
                parts.append(f"detectable: under {r['mde_ratio_under']:.2f}x / over {r['mde_ratio_over']:.2f}x")
            row.append("<br>".join(parts))
        hover.append(row)

    fig = go.Figure(go.Heatmap(
        z=z, x=cats, y=BRAND_KEYS, colorscale="RdBu_r", zmid=0,
        text=txt.values, texttemplate="%{text}", customdata=np.array(hover, dtype=object),
        hovertemplate="%{customdata}<extra></extra>", colorbar=dict(title="log2(ratio)")))
    fig.update_layout(height=400)
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"패스: {TAG_STAT} (Bonferroni 유의와 leave-one-out robust 모두 통과). 취약: 유의하지만 특정 경쟁사 제외 시 소실(거울상). "
               f"소표본: n<{SMALL_N} 셀, 수축 추정값 표시. 미검정: n<5. 대칭 수집 서브셋. "
               "판정 룰: docs/brand_category_test_rules.md")

    tgt = s[s["brand"] == TARGET_BRAND]
    name = BRANDS[TARGET_BRAND]["name_kr"]
    rob = tgt[tgt["robust"]].sort_values("ratio", ascending=False)
    frag = tgt[tgt["significant"] & ~tgt["robust"]]
    lines = []
    if not rob.empty:
        lines.append(f"**{TAG_STAT}** " + ", ".join(f"{r['topic_label']} {r['ratio']:.2f}x" for _, r in rob.iterrows()))
    if not frag.empty:
        lines.append("취약: " + ", ".join(f"{r['topic_label']} {r['ratio']:.2f}x ({r['loo_fragile_without']} 제외 시 소실)"
                                       for _, r in frag.iterrows()))
    if seg == "web_customer":
        und = tgt[~tgt["significant"] & (tgt["note"] == "")]
        if not und.empty:
            items = []
            for _, r in und.iterrows():
                if int(r["n_brand_cat"]) < SMALL_N:
                    items.append(f"{r['topic_label']} 미확정 (n={int(r['n_brand_cat'])}, 수축 추정 {r['ratio_shrunk']:.2f}x)")
                else:
                    items.append(f"{r['topic_label']} {r['ratio']:.2f}x, 검출 범위 {r['mde_ratio_under']:.2f}-{r['mde_ratio_over']:.2f}x 안")
            lines.append("범위 내, 미확정: " + ", ".join(items))
    if lines:
        st.info(f"{name} ({seg}, n={int(tgt['n_brand'].iloc[0]) if not tgt.empty else 0}):  \n" + "  \n".join(lines))
    else:
        st.info(f"{name}: 이 세그먼트에 유의 셀 없음")


def render():
    matrix, test, perf, lag = load()

    render_category_test(test)

    st.markdown("---")

    # --- Signal matrix
    st.subheader("Engagement vs Search Interest Signal Matrix")
    st.caption(f"YouTube 댓글 기반 engagement x 게시 익주 검색 WoW. [{TAG_CLUSTER}], 인과 아님")
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
    st.caption(f"MSTL 잔차 기반. VoC 볼륨 vs 검색 관심도. [{TAG_QUAL}], 작성자 미구분 볼륨, 광고 물량 교란 가능성으로 해석 보류")
    if lag:
        b = st.selectbox("Brand", list(lag.keys()), index=list(lag.keys()).index(TARGET_BRAND) if TARGET_BRAND in lag else 0)
        ld = pd.DataFrame([{"lag": int(k), "r": v} for k, v in lag[b].items()]).sort_values("lag")
        f = px.bar(ld, x="lag", y="r", color="r", color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                   labels={"lag": "Lag (weeks, + = VoC leads search)", "r": "Correlation"})
        for y, c in [(0.3, "green"), (-0.3, "red")]:
            f.add_hline(y=y, line_dash="dot", line_color=c, opacity=0.5)
        f.update_layout(height=320, coloraxis_showscale=False)
        st.plotly_chart(f, use_container_width=True)
