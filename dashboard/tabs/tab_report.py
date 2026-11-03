"""Tab 4: Weekly Report. AI 생성 리포트 미리보기와 PDF 다운로드."""

import os
import sys
import glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st
from config.settings import BRANDS, TARGET_BRAND
from config.tags import TAG_STAT, TAG_CLUSTER, TAG_QUAL, TAG_LOWN


def load_pdfs():
    return sorted(glob.glob("data/exports/cx_weekly_report_*.pdf"), reverse=True)


def load_report_text():
    try:
        from config.db import get_conn, _use_csv_fallback
        if _use_csv_fallback():
            return pd.DataFrame()
        with get_conn() as conn:
            return pd.read_sql(
                "SELECT brand, week_label, report_text, generated_at FROM mart.weekly_report "
                "ORDER BY generated_at DESC LIMIT 20", conn)
    except Exception:
        return pd.DataFrame()


def render():
    st.subheader(f"{BRANDS[TARGET_BRAND]['name_kr']} AI Weekly CX Report")
    st.caption(f"Claude Sonnet 생성. 시그널 태그: [{TAG_STAT}] 단정 가능, [{TAG_CLUSTER}] 패턴, "
               f"[{TAG_QUAL}] 참고, [{TAG_LOWN}] 미확정")

    pdfs = load_pdfs()
    if not pdfs:
        st.warning("No reports yet. Run `python -m report.weekly_report`.")
        return

    names = [os.path.basename(p) for p in pdfs]
    sel = st.selectbox("Select Report", names)
    path = pdfs[names.index(sel)]
    with open(path, "rb") as f:
        st.download_button("📥 Download PDF", f.read(), file_name=sel, mime="application/pdf")

    st.markdown("---")
    week_label = sel.replace("cx_weekly_report_", "").split("_")[0]
    df = load_report_text()
    if not df.empty:
        m = df[df["week_label"] == week_label]
        if not m.empty:
            st.markdown(f"**Generated:** {m.iloc[0]['generated_at']}")
            st.markdown("---")
            st.markdown(m.iloc[0]["report_text"])
            return
    # CSV fallback: show text from a sidecar .md if present
    md = path.replace(".pdf", ".md")
    if os.path.exists(md):
        st.markdown(open(md, encoding="utf-8").read())
    else:
        st.info("리포트 본문 미리보기는 DB 연결 시 표시됩니다. PDF를 다운로드해 확인하세요.")
