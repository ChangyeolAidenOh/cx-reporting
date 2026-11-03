"""Insurance CX Monitoring Dashboard.

Usage:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from config.settings import BRANDS, TARGET_BRAND
from config.tags import TAG_STAT, TAG_CLUSTER, TAG_QUAL, TAG_LOWN

st.set_page_config(page_title="Insurance CX Monitor", page_icon="📊", layout="wide")

target_kr = BRANDS[TARGET_BRAND]["name_kr"]
others = ", ".join(v["name_kr"] for k, v in BRANDS.items() if k != TARGET_BRAND)

st.title("손해보험 VoC와 브랜드 시그널 모니터링")
st.caption(f"Target: {target_kr} | Competitors: {others} | PoC v2.1")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Brand Tracker", "📊 Content & Category", "💬 VoC Monitor", "📄 Weekly Report",
])

with tab1:
    from dashboard.tabs.tab_brand import render
    render()
with tab2:
    from dashboard.tabs.tab_content import render
    render()
with tab3:
    from dashboard.tabs.tab_voc import render
    render()
with tab4:
    from dashboard.tabs.tab_report import render
    render()

with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "손해보험 5사의 공개 VoC, 검색 관심도, 공식 콘텐츠를 수집해 "
        "토픽 분류, 작성자 분류, 감성 분석, 소스와 작성자로 층화한 통계 검정을 거쳐 "
        "주간 CX 리포트를 자동 생성하는 PoC입니다."
    )
    st.markdown("---")
    st.markdown("**Signal tags**")
    st.markdown(
        f"- `{TAG_STAT}`: Bonferroni 유의 + leave-one-out robust\n"
        f"- `{TAG_CLUSTER}`: 패턴 관찰, 단정하지 않음\n"
        f"- `{TAG_QUAL}`: 원문 인용 등 참고용\n"
        f"- `{TAG_LOWN}`: 표본이 작아 미확정"
    )
    st.markdown("판정 룰: docs/brand_category_test_rules.md")
    st.markdown("---")
    st.markdown("**Data Sources**\n- Naver DataLab API\n- YouTube Data API v3\n- Naver Search API (Blog/Cafe)")
    st.markdown("---")
    st.caption("Independent Project | Changyeol Oh")
