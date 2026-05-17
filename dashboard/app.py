"""Insurance CX Monitoring Dashboard.

4-tab Streamlit dashboard:
  1. Brand Tracker: 4사 검색 관심도 추세, WoW, SoV
  2. Content Performance: 토픽별 Engagement vs 검색 관심도 매트릭스
  3. VoC Monitor: 감성 추이, 토픽별 감성, 이상 이벤트
  4. Weekly Report: AI 생성 리포트 미리보기 + PDF 다운로드

Usage:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(
    page_title="Insurance CX Monitor",
    page_icon="📊",
    layout="wide",
)

st.title("Insurance CX Weekly Reporting Dashboard")
st.caption("MetLife · Samsung Life · Hanwha Life · Kyobo Life | PoC v1.0")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Brand Tracker",
    "📊 Content Performance",
    "💬 VoC Monitor",
    "📄 Weekly Report",
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

# Sidebar
with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "보험사 4사 SNS 콘텐츠 성과를 모니터링하고, "
        "Engagement와 검색 관심도 간 패턴을 탐색하는 PoC 대시보드입니다."
    )
    st.markdown("---")
    st.markdown(
        "**Data Sources**\n"
        "- Naver DataLab API\n"
        "- YouTube Data API v3\n"
        "- Naver Search API (Blog/Cafe)"
    )
    st.markdown("---")
    st.caption("Independent Project | Changyeol Oh")
