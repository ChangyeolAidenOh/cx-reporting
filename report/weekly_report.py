"""AI Weekly CX Report Generator.

Core deliverable: 1-page MetLife CX Weekly Report PDF.
Queries mart tables -> structures data -> Claude Sonnet narrative -> PDF.

Usage:
    python -m report.weekly_report
    python -m report.weekly_report --week 2026-W20
    python -m report.weekly_report --brand metlife
"""

import sys
import os
import re
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from dotenv import load_dotenv
from config.db import get_conn

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LLM_MODEL_REPORT = os.getenv("LLM_MODEL_REPORT", "claude-sonnet-4-20250514")


# ============================================================
# Data gathering
# ============================================================

def get_latest_week():
    """Get the latest available week from mart data."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(week_start) FROM mart.brand_search_weekly")
            result = cur.fetchone()
    return result[0] if result else None


def gather_report_data(target_week, brand="metlife"):
    """Gather all data needed for the weekly report."""
    with get_conn() as conn:
        # 1. Brand search metrics
        search = pd.read_sql(
            f"""
            SELECT brand, week_start, search_ratio, wow_change_pct, sov_pct
            FROM mart.brand_search_weekly
            WHERE week_start = %s
            ORDER BY sov_pct DESC
            """,
            conn,
            params=(target_week,),
        )

        # Previous week for comparison
        prev_week = target_week - timedelta(days=7)
        search_prev = pd.read_sql(
            """
            SELECT brand, search_ratio, sov_pct
            FROM mart.brand_search_weekly
            WHERE week_start = %s
            """,
            conn,
            params=(prev_week,),
        )

        # 2. Content performance
        content = pd.read_sql(
            """
            SELECT brand, source, topic_label, content_count,
                   avg_engagement, total_views, total_likes, total_comments
            FROM mart.content_performance
            WHERE week_start = %s
            """,
            conn,
            params=(target_week,),
        )

        # 3. VoC sentiment
        # To get dataset of VoC, retain recent 4-week rolling window as samples
        sentiment_start = target_week - timedelta(days=28)
        sentiment = pd.read_sql(
            """
            SELECT brand,
                   AVG(positive_ratio) AS positive_ratio,
                   AVG(neutral_ratio) AS neutral_ratio,
                   AVG(negative_ratio) AS negative_ratio,
                   SUM(total_docs) AS total_docs,
                   COUNT(*) AS weeks_covered
            FROM mart.voc_sentiment
            WHERE week_start BETWEEN %s AND %s
            GROUP BY brand
            """,
            conn,
            params=(sentiment_start, target_week),
        )

        # 4. Anomaly events (recent 4 weeks)
        anomaly_start = target_week - timedelta(days=28)
        anomalies = pd.read_sql(
            """
            SELECT brand, detected_date, metric, value, zscore,
                   methods_agreed, probable_cause
            FROM mart.anomaly_log
            WHERE detected_date BETWEEN %s AND %s
            ORDER BY detected_date DESC
            """,
            conn,
            params=(anomaly_start, target_week),
        )

        # 5. Signal matrix
        signals = pd.read_sql(
            "SELECT * FROM mart.content_signal_matrix",
            conn,
        )

        # 6. Lead-lag results
        lag_path = "data/exports/lead_lag_results.json"
        lag_results = {}
        if os.path.exists(lag_path):
            with open(lag_path, "r") as f:
                lag_results = json.load(f)

    return {
        "search": search,
        "search_prev": search_prev,
        "content": content,
        "sentiment": sentiment,
        "anomalies": anomalies,
        "signals": signals,
        "lag_results": lag_results,
        "target_week": target_week,
        "brand": brand,
    }


# ============================================================
# LLM narrative generation
# ============================================================

def build_prompt(data):
    """Build structured prompt for Claude Sonnet."""
    brand = data["brand"]
    week = data["target_week"]
    week_label = f"{week.year}-W{week.isocalendar()[1]:02d}"

    # Format data sections
    search_summary = ""
    if not data["search"].empty:
        brand_map = {"삼성생명": "samsung", "한화생명": "hanwha", "교보생명": "kyobo", "메트라이프": "metlife"}
        for _, row in data["search"].iterrows():
            brand_key = brand_map.get(row["brand"], row["brand"])
            marker = " ← TARGET" if brand_key == brand else ""
            wow = f"{row['wow_change_pct']:+.1f}%" if pd.notna(row["wow_change_pct"]) else "N/A"
            search_summary += f"  {row['brand']}: SoV {row['sov_pct']:.1f}%, WoW {wow}{marker}\n"

    sentiment_summary = ""
    if not data["sentiment"].empty:
        for _, row in data["sentiment"].iterrows():
            if row["brand"] == brand:
                weeks = int(row.get("weeks_covered", 1))
                sentiment_summary += f"  (최근 {weeks}주 롤링 평균)\n"
                sentiment_summary += (
                    f"  Positive: {row['positive_ratio']:.1%}, "
                    f"Neutral: {row['neutral_ratio']:.1%}, "
                    f"Negative: {row['negative_ratio']:.1%}\n"
                    f"  Total docs: {int(row['total_docs'])}\n"
                )


    anomaly_summary = "없음"
    if not data["anomalies"].empty:
        brand_anomalies = data["anomalies"][
            data["anomalies"]["brand"].str.contains(brand, case=False, na=False) |
            data["anomalies"]["brand"].map(
                {"삼성생명": "samsung", "한화생명": "hanwha", "교보생명": "kyobo", "메트라이프": "metlife"}
            ).eq(brand)
        ]
        if not brand_anomalies.empty:
            lines = []
            for _, a in brand_anomalies.iterrows():
                lines.append(f"  {a['detected_date']}: {a['metric']}={a['value']:.1f} (z={a['zscore']:.2f})")
            anomaly_summary = "\n".join(lines)

    lag_summary = ""
    if brand in data["lag_results"]:
        lags = data["lag_results"][brand]
        best_lag = max(lags, key=lambda k: abs(lags[k]))
        lag_summary = f"  Strongest: lag={best_lag}, r={lags[best_lag]:.4f}"

    prompt = f"""당신은 보험사 CX 분석 전문가입니다.
아래 데이터를 바탕으로 {week_label} MetLife CX 주간 리포트를 작성해주세요.

지침:
- 한국어로 작성
- 간결하고 실무적인 톤
- 인과 관계를 단정하지 않고 "패턴이 관찰됨", "연관 가능성" 등의 표현 사용
- 원인 후보가 있을 경우 콘텐츠/뉴스/캠페인/기타 중 명시
- 5개 섹션 구조 유지

=== business_context  ===
MetLife Korea는 프로지점제(설계사 중심 유통) + 변액/달러보험(니치 고가 상품) 중심의 사업 구조.
국내 대형사(삼성/한화/교보)와 달리 대중 시장 도달(reach)보다 기존 고객 유지(retention)와
설계사 지원용 전문성 콘텐츠가 구조적으로 적합.
SoV 5.2%는 브랜딩 실패가 아닌 사업 모델 반영.
리포트 작성 시 이 맥락을 반영하여, 검색 SoV 확대보다 콘텐츠 효율(content efficiency)과
360Health/360Future 등 기존 고객 서비스와의 연결 관점에서 해석해주세요.

[1. 브랜드 검색 관심도]
{search_summary}

[2. VoC 감성 동향]
{sentiment_summary if sentiment_summary else "해당 주 데이터 부족"}

[3. 이상 이벤트]
{anomaly_summary}

[4. Lead-Lag 참고 신호]
{lag_summary if lag_summary else "분석 중"}

=== 출력 형식 ===

MetLife CX Weekly Report — {week_label}

1. 콘텐츠 성과 요약
   (게시 건수, 유형별 분포, 주요 성과 콘텐츠)

2. 브랜드 검색 관심도
   (MetLife 검색 변화, 경쟁사 대비 SoV, 변화 원인 후보)

3. VoC 감성 동향
   (긍정/중립/부정 비율, 주요 키워드)

4. 이상 이벤트
   (있을 경우만 기술)

5. 참고 신호
   (lead-lag 패턴, 데이터 한계 명시)
"""

    return prompt


def generate_narrative(prompt):
    """Call Claude Sonnet to generate report narrative."""
    try:
        import anthropic
    except ImportError:
        print("[ERROR] anthropic not installed")
        return None

    if not ANTHROPIC_API_KEY:
        print("[ERROR] ANTHROPIC_API_KEY not set")
        return None

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    response = client.messages.create(
        model=LLM_MODEL_REPORT,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text


# ============================================================
# PDF generation
# ============================================================

def generate_pdf(narrative, week_label, brand, output_dir="data/exports"):
    """Generate 1-page PDF report using ReportLab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/cx_weekly_report_{week_label}_{brand}.pdf"

    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
    )

    styles = getSampleStyleSheet()

    # Try to register Korean font
    korean_font = None
    font_paths = [
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/Library/Fonts/NanumGothic.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                pdfmetrics.registerFont(TTFont("KoreanFont", fp))
                korean_font = "KoreanFont"
                break
            except Exception:
                continue

    base_font = korean_font or "Helvetica"

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName=base_font,
        fontSize=16,
        textColor=HexColor("#1a5276"),
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName=base_font,
        fontSize=9,
        textColor=HexColor("#7f8c8d"),
        spaceAfter=12,
    )

    heading_style = ParagraphStyle(
        "SectionHead",
        parent=styles["Heading2"],
        fontName=base_font,
        fontSize=11,
        textColor=HexColor("#2c3e50"),
        spaceBefore=10,
        spaceAfter=4,
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName=base_font,
        fontSize=9,
        leading=14,
        spaceAfter=6,
    )

    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName=base_font,
        fontSize=7,
        textColor=HexColor("#95a5a6"),
        spaceBefore=12,
    )

    # Build story
    story = []
    story.append(Paragraph(f"MetLife CX Weekly Report — {week_label}", title_style))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | AI-assisted analysis",
        subtitle_style,
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor("#1a5276")))
    story.append(Spacer(1, 6))

    # Parse narrative into sections
    sections = re.split(r"\n(?=\d+\.)", narrative)
    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Try to split heading from body
        lines = section.split("\n", 1)
        heading = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""

        if heading and heading[0].isdigit():
            story.append(Paragraph(heading, heading_style))
        else:
            story.append(Paragraph(heading, body_style))

        if body:
            # Replace newlines with <br/> for PDF
            body_html = body.replace("\n", "<br/>")
            body_html = body_html.replace("  ", "&nbsp;&nbsp;")
            story.append(Paragraph(body_html, body_style))

    # Disclaimer
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#bdc3c7")))
    story.append(Paragraph(
        "본 리포트는 공개 데이터 기반 AI 분석 결과이며, 인과 관계를 확정하지 않습니다. "
        "콘텐츠 유형 변경 시 준법감시 프로세스 리드타임을 고려해주세요.",
        disclaimer_style,
    ))

    doc.build(story)
    print(f"PDF generated: {filename}")
    return filename


# ============================================================
# Save to DB
# ============================================================

def save_report_to_db(brand, week_label, report_data, narrative, pdf_path):
    """Save report metadata to mart.weekly_report."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Serialize report_data (remove DataFrames)
            report_json = {
                "search": report_data["search"].to_dict(orient="records") if not report_data["search"].empty else [],
                "anomalies": report_data["anomalies"].to_dict(orient="records") if not report_data["anomalies"].empty else [],
                "lag_results": report_data.get("lag_results", {}),
            }

            cur.execute(
                """
                INSERT INTO mart.weekly_report (brand, week_label, report_json, report_text, pdf_path)
                VALUES (%s, %s, %s::jsonb, %s, %s)
                ON CONFLICT (brand, week_label)
                DO UPDATE SET report_json = EXCLUDED.report_json,
                              report_text = EXCLUDED.report_text,
                              pdf_path = EXCLUDED.pdf_path,
                              generated_at = NOW()
                """,
                (
                    brand, week_label,
                    json.dumps(report_json, ensure_ascii=False, default=str),
                    narrative, pdf_path,
                ),
            )
    print(f"Report saved to mart.weekly_report")


# ============================================================
# Main
# ============================================================

def run(week=None, brand="metlife"):
    print("=" * 70)
    print("AI Weekly CX Report Generator")
    print("=" * 70)

    # Determine target week
    if week:
        year, w = week.split("-W")
        target_week = datetime.strptime(f"{year}-W{w}-1", "%Y-W%W-%w").date()
    else:
        target_week = get_latest_week()
        if not target_week:
            print("[ERROR] No data in mart tables")
            return

    week_label = f"{target_week.year}-W{target_week.isocalendar()[1]:02d}"
    print(f"Target: {brand} | Week: {week_label} ({target_week})")

    # Step 1: Gather data
    print("\n1. Gathering report data...")
    data = gather_report_data(target_week, brand)

    # Step 2: Generate narrative
    print("\n2. Generating AI narrative...")
    prompt = build_prompt(data)
    narrative = generate_narrative(prompt)

    if not narrative:
        print("[ERROR] Failed to generate narrative")
        return

    print(f"  Narrative generated ({len(narrative)} chars)")

    # Step 3: Generate PDF
    print("\n3. Generating PDF...")
    pdf_path = generate_pdf(narrative, week_label, brand)

    # Step 4: Save to DB
    print("\n4. Saving to database...")
    save_report_to_db(brand, week_label, data, narrative, pdf_path)

    # Print narrative preview
    print("\n" + "=" * 70)
    print("REPORT PREVIEW")
    print("=" * 70)
    print(narrative)

    print(f"\nPhase 5 complete.")
    print(f"  PDF: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate weekly CX report")
    parser.add_argument("--week", default=None, help="Target week (e.g., 2026-W20)")
    parser.add_argument("--brand", default="metlife", help="Target brand")
    args = parser.parse_args()

    run(week=args.week, brand=args.brand)


if __name__ == "__main__":
    main()
