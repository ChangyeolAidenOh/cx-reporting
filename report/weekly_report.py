"""AI Weekly CX Report Generator.

Core deliverable: 1-page weekly CX report PDF for the target brand.
Queries mart tables -> structured prompt with signal-strength tags
-> Claude Sonnet narrative -> PDF.

Usage:
    python -m report.weekly_report
    python -m report.weekly_report --week 2026-W44
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
from config.settings import BRANDS, BRAND_KR_TO_KEY, TARGET_BRAND

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LLM_MODEL_REPORT = os.getenv("LLM_MODEL_REPORT", "claude-sonnet-4-6")

TARGET_KR = BRANDS[TARGET_BRAND]["name_kr"]
KEY_TO_KR = {k: v["name_kr"] for k, v in BRANDS.items()}

BUSINESS_CONTEXT = f"""{TARGET_KR}은 KB금융그룹 계열 손해보험사로, 자동차보험·장기(실손·건강)보험·일반보험을
다이렉트 채널과 설계사 채널 양쪽으로 판매한다. 검색 관심도 기준 삼성화재·현대해상·DB손해보험과
20% 초반대 SoV로 접전 중인 빅4 구도이며, 메리츠화재가 그 아래에 있다.
손해보험 CX의 핵심 접점은 보험금 청구·보상 처리 경험과 자동차보험 갱신 시점이다.
이 리포트는 데이터 기반 의사결정 지원 목적이므로, 모든 수치에 시그널 강도 태그를 붙이고
인과 관계는 단정하지 않는다."""


# ============================================================
# Data gathering
# ============================================================

def get_latest_week():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(week_start) FROM mart.brand_search_weekly")
            r = cur.fetchone()
    return r[0] if r else None


def gather(target_week):
    prev_week = target_week - timedelta(days=7)
    win_start = target_week - timedelta(days=28)
    anom_start = target_week - timedelta(days=28)

    with get_conn() as conn:
        search = pd.read_sql(
            "SELECT brand, search_ratio, wow_change_pct, sov_pct FROM mart.brand_search_weekly "
            "WHERE week_start = %s ORDER BY sov_pct DESC", conn, params=(target_week,))
        sentiment = pd.read_sql(
            """
            SELECT brand, AVG(positive_ratio) AS positive_ratio, AVG(neutral_ratio) AS neutral_ratio,
                   AVG(negative_ratio) AS negative_ratio, SUM(total_docs) AS total_docs,
                   COUNT(*) AS weeks_covered
            FROM mart.voc_sentiment WHERE week_start BETWEEN %s AND %s GROUP BY brand
            """, conn, params=(win_start, target_week))
        anomalies = pd.read_sql(
            "SELECT brand, detected_date, metric, value, zscore, methods_agreed FROM mart.anomaly_log "
            "WHERE detected_date BETWEEN %s AND %s ORDER BY detected_date DESC",
            conn, params=(anom_start, target_week))
        reps = pd.read_sql(
            "SELECT topic_label, sentiment_label, text_clean FROM mart.topic_representatives "
            "WHERE brand = %s ORDER BY topic_label, sentiment_label", conn, params=(TARGET_BRAND,))
        content = pd.read_sql(
            "SELECT brand, content_count, avg_engagement, total_views, total_comments "
            "FROM mart.content_performance WHERE week_start = %s", conn, params=(target_week,))

    cat_test_path = "data/exports/brand_category_test.csv"
    cat_test = pd.read_csv(cat_test_path) if os.path.exists(cat_test_path) else pd.DataFrame()
    lag_path = "data/exports/lead_lag_results.json"
    lag = json.load(open(lag_path)) if os.path.exists(lag_path) else {}

    return dict(target_week=target_week, search=search, sentiment=sentiment, anomalies=anomalies,
                reps=reps, content=content, cat_test=cat_test, lag=lag)


# ============================================================
# Prompt
# ============================================================

def fmt_search(d):
    lines = []
    for _, r in d["search"].iterrows():
        key = BRAND_KR_TO_KEY.get(r["brand"], r["brand"])
        mark = " <- TARGET" if key == TARGET_BRAND else ""
        wow = f"{r['wow_change_pct']:+.1f}%" if pd.notna(r["wow_change_pct"]) else "N/A"
        lines.append(f"  {r['brand']}: SoV {r['sov_pct']:.1f}%, WoW {wow}{mark}  [통계 검증 ✓: 관측값]")
    return "\n".join(lines)


def fmt_content(d):
    c = d["content"]
    c = c[c["brand"] == TARGET_KR]
    if c.empty:
        return "  해당 주 공식 YouTube 신규 게시 없음"
    r = c.iloc[0]
    return (f"  신규 영상 {int(r['content_count'])}건, 조회 {int(r['total_views']):,}, "
            f"댓글 {int(r['total_comments'])}, engagement {r['avg_engagement']:.3f}%  [cluster 시그널]")


def fmt_cat_test(d):
    t = d["cat_test"]
    if t.empty:
        return "  분석 중"
    tgt = t[(t["brand"] == TARGET_BRAND) & t["significant"]].sort_values("ratio", ascending=False)
    lines = []
    for _, r in tgt.iterrows():
        lines.append(f"  {r['topic_label']}: {r['ratio']:.2f}x (경쟁 4사 평균 대비) "
                     f"[{r['signal_tag']}, Bonferroni 보정]")
    lines.append("  ※ 대칭 수집 서브셋 기준 (타겟 브랜드 상품별 쿼리 제외)")
    return "\n".join(lines)


def fmt_sentiment(d):
    s = d["sentiment"]
    s = s[s["brand"] == TARGET_BRAND]
    if s.empty:
        return "  데이터 부족"
    r = s.iloc[0]
    return (f"  (최근 {int(r['weeks_covered'])}주 롤링) Positive {r['positive_ratio']:.1%}, "
            f"Neutral {r['neutral_ratio']:.1%}, Negative {r['negative_ratio']:.1%}, "
            f"n={int(r['total_docs'])}  [cluster 시그널]")


def fmt_reps(d, n_neg=3, n_pos=2):
    reps = d["reps"]
    if reps.empty:
        return "  없음"
    out = []
    neg = reps[reps["sentiment_label"] == "negative"].head(n_neg)
    pos = reps[reps["sentiment_label"] == "positive"].head(n_pos)
    for _, r in neg.iterrows():
        out.append(f"  [부정·{r['topic_label']}] \"{r['text_clean'][:110]}\"")
    for _, r in pos.iterrows():
        out.append(f"  [긍정·{r['topic_label']}] \"{r['text_clean'][:110]}\"")
    out.append("  ※ 원문 어휘 그대로 인용 가능 (변형 시 정합도 손실)  [정성 시그널]")
    return "\n".join(out)


def fmt_anomalies(d):
    a = d["anomalies"]
    a = a[a["brand"] == TARGET_KR]
    if a.empty:
        return "  없음"
    return "\n".join(f"  {r['detected_date']}: {r['metric']}={r['value']:.1f} "
                     f"(z={r['zscore']:+.2f}, {int(r['methods_agreed'])}/3 methods)  [통계 검증 ✓]"
                     for _, r in a.iterrows())


def fmt_lag(d):
    lags = d["lag"].get(TARGET_BRAND, {})
    if not lags:
        return "  분석 중"
    best = max(lags, key=lambda k: abs(lags[k]))
    return (f"  최강 시차 lag={int(best):+d}주, r={lags[best]:+.3f}  [정성 시그널: VoC 볼륨 기반, "
            f"광고 물량 교란 가능성 있어 해석 보류]")


def build_prompt(d):
    week = d["target_week"]
    week_label = f"{week.year}-W{week.isocalendar()[1]:02d}"
    return f"""당신은 손해보험사 데이터 분석가입니다.
아래 데이터를 바탕으로 {week_label} {TARGET_KR} CX 주간 리포트를 작성하세요.

지침:
- 한국어, 간결하고 실무적인 톤, 1페이지 분량
- 모든 수치 옆의 시그널 태그를 리포트에 그대로 유지: [통계 검증 ✓] [cluster 시그널] [정성 시그널]
- [통계 검증 ✓]만 단정 표현 허용. 나머지는 "패턴 관찰", "연관 가능성" 표현
- 인과 관계 단정 금지. 원인 후보는 콘텐츠/뉴스/캠페인/시즌/기타로 구분
- 대표 VoC 인용은 어휘 변형 없이 그대로
- 6개 섹션 구조 유지

=== Business Context ===
{BUSINESS_CONTEXT}

=== 데이터 ===

[1. 브랜드 검색 관심도]
{fmt_search(d)}

[2. 공식 콘텐츠 성과]
{fmt_content(d)}

[3. VoC 카테고리 구성 — 경쟁사 대비 (브랜드×카테고리 비중 검정)]
{fmt_cat_test(d)}

[4. VoC 감성]
{fmt_sentiment(d)}

[5. 대표 VoC]
{fmt_reps(d)}

[6. 이상 이벤트 (최근 4주)]
{fmt_anomalies(d)}

[7. Lead-Lag 참고]
{fmt_lag(d)}

=== 출력 형식 ===

{TARGET_KR} CX Weekly Report — {week_label}

1. 핵심 요약 (3줄 이내)
2. 브랜드 검색 관심도 (SoV·WoW, 경쟁사 대비, 원인 후보)
3. VoC 구성·감성 (카테고리 과대/과소 인덱싱 해석, 감성 비율)
4. 대표 고객 목소리 (부정 우선, 원문 인용)
5. 이상 이벤트
6. 참고 신호·데이터 한계 (lead-lag, 표본, 수집 편향 단서)
"""


# ============================================================
# LLM + PDF
# ============================================================

def generate_narrative(prompt):
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    resp = client.messages.create(model=LLM_MODEL_REPORT, max_tokens=2500,
                                  messages=[{"role": "user", "content": prompt}])
    return resp.content[0].text


def generate_pdf(narrative, week_label, output_dir="data/exports"):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/cx_weekly_report_{week_label}_{TARGET_BRAND}.pdf"
    doc = SimpleDocTemplate(filename, pagesize=A4, topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                            leftMargin=2 * cm, rightMargin=2 * cm)

    font = "Helvetica"
    for fp in ["/System/Library/Fonts/Supplemental/AppleGothic.ttf",
               "/Library/Fonts/NanumGothic.ttf"]:
        if os.path.exists(fp):
            try:
                pdfmetrics.registerFont(TTFont("KoreanFont", fp))
                font = "KoreanFont"
                break
            except Exception:
                pass

    ss = getSampleStyleSheet()
    st_title = ParagraphStyle("t", parent=ss["Title"], fontName=font, fontSize=15,
                              textColor=HexColor("#5a4b00"), spaceAfter=4)
    st_sub = ParagraphStyle("s", parent=ss["Normal"], fontName=font, fontSize=8.5,
                            textColor=HexColor("#7f8c8d"), spaceAfter=10)
    st_h = ParagraphStyle("h", parent=ss["Heading2"], fontName=font, fontSize=10.5,
                          textColor=HexColor("#2c3e50"), spaceBefore=8, spaceAfter=3)
    st_b = ParagraphStyle("b", parent=ss["Normal"], fontName=font, fontSize=8.5,
                          leading=13, spaceAfter=5)
    st_d = ParagraphStyle("d", parent=ss["Normal"], fontName=font, fontSize=7,
                          textColor=HexColor("#95a5a6"), spaceBefore=10)

    story = [Paragraph(f"{TARGET_KR} CX Weekly Report — {week_label}", st_title),
             Paragraph(f"Generated {datetime.now():%Y-%m-%d %H:%M} | AI-assisted | "
                       f"태그: [통계 검증 ✓] 단정 가능 · [cluster 시그널] 패턴 · [정성 시그널] 참고", st_sub),
             HRFlowable(width="100%", thickness=1, color=HexColor("#5a4b00")), Spacer(1, 4)]

    clean = re.sub(r"^#+\s*", "", narrative, flags=re.M).replace("**", "")
    for sec in re.split(r"\n(?=\d+\.\s)", clean):
        sec = sec.strip()
        if not sec:
            continue
        head, _, body = sec.partition("\n")
        story.append(Paragraph(head.strip(), st_h if head[0].isdigit() else st_b))
        if body.strip():
            story.append(Paragraph(body.strip().replace("\n", "<br/>"), st_b))

    story += [Spacer(1, 8), HRFlowable(width="100%", thickness=0.5, color=HexColor("#bdc3c7")),
              Paragraph("본 리포트는 공개 데이터 기반 AI 분석이며 인과 관계를 확정하지 않습니다. "
                        "브랜드 간 비교는 대칭 수집 서브셋 기준이고, 타겟 브랜드 심층 VoC는 별도 수집분을 포함합니다.", st_d)]
    doc.build(story)
    print(f"PDF generated: {filename}")
    return filename


def save_report(week_label, d, narrative, pdf_path):
    payload = {"search": d["search"].to_dict(orient="records"),
               "anomalies": d["anomalies"].to_dict(orient="records"),
               "lag": d["lag"].get(TARGET_BRAND, {})}
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO mart.weekly_report (brand, week_label, report_json, report_text, pdf_path)
                VALUES (%s, %s, %s::jsonb, %s, %s)
                ON CONFLICT (brand, week_label) DO UPDATE SET
                    report_json = EXCLUDED.report_json, report_text = EXCLUDED.report_text,
                    pdf_path = EXCLUDED.pdf_path, generated_at = NOW()
                """,
                (TARGET_BRAND, week_label, json.dumps(payload, ensure_ascii=False, default=str),
                 narrative, pdf_path))
    print("Report saved to mart.weekly_report")


# ============================================================
# Main
# ============================================================

def run(week=None):
    if week:
        y, w = week.split("-W")
        target = datetime.strptime(f"{y}-W{w}-1", "%Y-W%W-%w").date()
    else:
        target = get_latest_week()
    week_label = f"{target.year}-W{target.isocalendar()[1]:02d}"
    print(f"Target: {TARGET_KR} | Week: {week_label} ({target})")

    d = gather(target)
    prompt = build_prompt(d)
    narrative = generate_narrative(prompt)
    print(f"Narrative generated ({len(narrative)} chars)")
    pdf = generate_pdf(narrative, week_label)
    save_report(week_label, d, narrative, pdf)
    print("\n" + "=" * 70 + "\nREPORT PREVIEW\n" + "=" * 70)
    print(narrative)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--week", default=None, help="e.g. 2026-W44")
    a = p.parse_args()
    run(week=a.week)


if __name__ == "__main__":
    main()
