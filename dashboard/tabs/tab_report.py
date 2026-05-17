"""Tab 4: Weekly Report - AI 생성 리포트 미리보기 + PDF 다운로드."""

import os
import glob
import pandas as pd
import streamlit as st


def load_reports():
    """Load available reports from exports."""
    pdf_dir = "data/exports"
    pdfs = sorted(glob.glob(f"{pdf_dir}/cx_weekly_report_*.pdf"), reverse=True)
    return pdfs


def load_report_text():
    """Load latest report text from DB or CSV."""
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from config.db import get_conn, _use_csv_fallback

        if not _use_csv_fallback():
            with get_conn() as conn:
                df = pd.read_sql(
                    "SELECT brand, week_label, report_text, generated_at "
                    "FROM mart.weekly_report ORDER BY generated_at DESC LIMIT 10",
                    conn,
                )
            return df
    except Exception:
        pass
    return pd.DataFrame()


def render():
    st.subheader("AI-Generated Weekly CX Report")

    # Load available PDFs
    pdfs = load_reports()

    if not pdfs:
        st.warning("No reports generated yet. Run `python -m report.weekly_report` to generate.")
        return

    # Report selector
    pdf_names = [os.path.basename(p) for p in pdfs]
    selected = st.selectbox("Select Report", pdf_names)
    selected_path = pdfs[pdf_names.index(selected)]

    # PDF download
    with open(selected_path, "rb") as f:
        pdf_bytes = f.read()

    st.download_button(
        label="📥 Download PDF",
        data=pdf_bytes,
        file_name=selected,
        mime="application/pdf",
    )

    st.markdown("---")

    # Report preview (from DB)
    reports_df = load_report_text()

    if not reports_df.empty:
        # Extract week from filename
        week_from_file = selected.replace("cx_weekly_report_", "").replace("_metlife.pdf", "").replace(".pdf", "")

        # Find matching report
        matching = reports_df[reports_df["week_label"].str.contains(week_from_file[:8], na=False)]
        if not matching.empty:
            report = matching.iloc[0]
            st.markdown(f"**Generated:** {report['generated_at']}")
            st.markdown("---")
            st.markdown(report["report_text"])
        else:
            st.info("Report text preview not available for this week.")
    else:
        st.info("Connect to database for report text preview, or view the downloaded PDF.")

    st.markdown("---")

    # Generation controls
    with st.expander("Generate New Report"):
        st.markdown("""
        To generate a new weekly report:
        ```bash
        python -m report.weekly_report
        python -m report.weekly_report --week 2026-W19
        python -m report.weekly_report --brand metlife
        ```
        """)
        st.caption("Report generation requires Claude API access.")
