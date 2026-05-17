# Insurance CX Weekly Reporting Automation

> **Position:** MetLife Korea — Internship, Branding & Customer Experience
> **Positioning:** Content Performance Monitoring Framework PoC + AI Weekly CX Report Automation Prototype

**Independent Project**

---

## Overview

Insurance companies measure SNS content performance primarily through engagement metrics (likes, comments, shares). But does high engagement actually correlate with brand consideration? This project explores that question by combining topic-classified content from 4 Korean insurers (MetLife, Samsung Life, Hanwha Life, Kyobo Life) with search interest signals, using a dual measurement framework: **Engagement Score × Search Interest Signal**.

The core deliverable is an **AI-generated 1-page weekly CX report** — a direct implementation of the JD requirement for "AI(Copilot) 활용 정기 Report 업무 효율화."

> **Boundary:** This project does not claim "content causes conversion." It explores lead-lag patterns between content types and brand search interest, presents findings as exploratory signals with alternative cause candidates, and automates the reporting of these patterns for weekly operational use.

---

## Key Findings

1. **Industry-wide co-movement detected:** In Week 2026-W20, all 4 insurers showed similar search volume decline (-32% to -37%), suggesting external factors rather than brand-specific issues. This pattern would be invisible in single-brand monitoring.

2. **MetLife content → search lag = +2 weeks (r=0.22):** The only brand showing a positive correlation between content volume and subsequent search interest. Weak but directionally meaningful — the report recommends monitoring W+2 search changes after content pushes.

3. **Hanwha's health campaign dominated engagement:** The "봉인하고 싶어요" (disease-sealing wish card) campaign generated 130+ comments across Topics 12 and 39, demonstrating that health/lifestyle content drives the highest engagement in the insurance vertical.

4. **MetLife VoC negative ratio (8.8%) is manageable** but with structural SoV disadvantage (5.2% vs. industry): The foreign brand's search volume is inherently lower, making content-driven search lifts both more important and harder to achieve.

5. **Engagement ≠ Search Interest:** 13 of 15 brand × topic combinations fell in the "high engagement, negative search signal" quadrant. Only Hanwha's health and education content showed positive search signal correlation.

6. **MetLife SoV 5.2% is structural, not a branding failure:** MetLife Korea operates a pro-branch system (프로지점제) with niche high-value products (variable/dollar insurance), fundamentally different from domestic insurers' mass-market model. Blog/Cafe VoC volume is structurally low because the customer journey bypasses public channels. This means content strategy should target retention and advisor-shareable expertise, not mass reach.

7. **Channel-dependent perception gap:** MetLife Blog VoC shows 8.8% negative, but App Store reviews show 72.7% negative (avg rating 1.9/5.0, n=88). The same brand appears healthy or critical depending on the VoC channel — validating the need for multi-channel monitoring. Blog VoC is dominated by advisor-written promotional content; App Store reviews capture actual user experience with the MetLife ONE digital platform.

8. **MetLife's real competitive set is foreign insurers, not domestic:** Co-mention analysis from 2,479 insurance-community VoC documents reveals MetLife is most frequently compared with Prudential (127 co-mentions), Chubb (96), and AIG (76) — not Samsung or Hanwha. This suggests the 4-brand domestic comparison captures industry-wide patterns, but MetLife-specific competitive strategy should reference the foreign insurer landscape.

---

## Pipeline

```
[Data Collection -- 5 collectors]
|-- Naver DataLab API: 4-brand search trends (weekly, 124 weeks)
|-- YouTube Data API v3: 3-brand videos + comments (MetLife excluded)
|-- Naver Search API: Blog/Cafe VoC for all 4 brands
|-- App Store (iTunes RSS): MetLife ONE app reviews
|-- Naver Search API: Foreign insurer community VoC
         |
         v
[ETL -- PostgreSQL 3-tier]
raw (6 tables) --> staging (1 table) --> mart (6 tables)
|-- raw.naver_datalab: 496 rows (4 brands x 124 weeks)
|-- raw.youtube_videos: 300 videos
|-- raw.youtube_comments: 5,149 comments
|-- raw.naver_voc: 7,020 + 2,479 (general + foreign insurer VoC)
|-- raw.app_reviews: 91 MetLife ONE App Store reviews
         |
         v
[NLP Analysis]
|-- BERTopic (ko-sroberta) x LDA ensemble: 43 topics discovered
|   |-- Manual mapping to 6 categories (see Decision Point 3)
|   |-- Noise re-inspection rescued 310 docs (see Decision Point 4)
|-- Hybrid sentiment analysis
|   |-- Layer 1: Insurance domain dictionary (37 pos + 28 neg terms)
|   |-- Layer 2: Claude Haiku API for ambiguous cases (2,860 docs)
|   |-- Routing: confidence < 0.3 triggers LLM call
         |
         v
[Signal Analysis]
|-- Engagement vs Search Interest matrix (15 brand x topic cells)
|-- Lead-lag cross-correlation (MSTL residual, ±8 week window)
|-- 3-way anomaly detection (Z-score + MSTL + Isolation Forest)
|   |-- 19 anomalies detected across 4 brands
         |
         v
[AI Report Generation -- Core Deliverable]
|-- Claude Sonnet: structured data --> 1-page narrative
|-- ReportLab: narrative --> PDF with Korean font support
|-- mart.weekly_report: report metadata + text stored in DB
         |
         v
[Dashboard -- Streamlit 4 tabs]
|-- Brand Tracker: SoV trend, WoW heatmap
|-- Content Performance: signal matrix scatter, lead-lag bar
|-- VoC Monitor: sentiment stacked area, brand comparison
|-- Weekly Report: PDF preview + download
```

---

## Business Context: MetLife vs Domestic Insurers

This project compares 4 insurers on equal terms, but their business models are fundamentally different:

| | Domestic (Samsung/Hanwha/Kyobo) | MetLife Korea |
|---|---|---|
| Distribution | Mass-market, direct insurance, TV ads | Pro-branch system (프로지점제, since 1998) |
| Core Products | Broad portfolio, consumer-friendly | Variable/dollar insurance (niche, high-value) |
| Content Goal | Reach (brand awareness) | Retention + advisor enablement |
| VoC Generation | Organic blog/cafe reviews | Structurally low — customer journey is advisor-mediated |

**Implication for this project:**
- The 4-brand comparison framework is valid for **industry-wide pattern detection** (e.g., the W20 co-decline)
- MetLife-specific content strategy requires a **separate lens**: content efficiency per touchpoint, not SoV volume
- **Next step:** Foreign insurer community VoC and MetLife ONE app reviews have been collected and analyzed, confirming the channel-dependent perception gap and MetLife's actual competitive set (see Findings 7, 8).

---

## Data Sources

| Priority | Data | Source | Volume | Stability |
|---|---|---|---|---|
| **Core** | Brand search trends | Naver DataLab API | 496 rows (124 weeks × 4 brands) | Stable |
| **Core** | YouTube videos + comments | YouTube Data API v3 | 300 videos, 5,149 comments | Stable |
| **Core** | Consumer VoC | Naver Search API (Blog/Cafe) | 7,020 unique docs | Stable |
| **Core** | MetLife ONE app reviews | App Store (iTunes RSS) | 91 reviews (avg 1.9★) | Stable |
| **Core** | Foreign insurer VoC | Naver Search API (targeted) | 2,479 unique docs | Stable |
| Supplementary | Instagram posts | Public profile | Not collected (API limitation) |  |

### Pre-EDA Data Volume Verification

Before full collection, three validation checks were run (`scripts/run_pre_eda.py`):

| Check | Result | Decision |
|---|---|---|
| MetLife Blog VoC monthly volume | 214 posts/month | Sufficient — proceed with Blog as primary VoC |
| YouTube official channels | MetLife: 31 subscribers (MDRT committee only) | No official channel — Blog VoC only for MetLife |
| Naver DataLab SoV | MetLife 5.2% vs Samsung 42.4% | Low but usable — foreign brand baseline |

---

## Methodology & Decision Points

### Decision Point 1: YouTube Channel Misidentification

The YouTube search API returned incorrect channels for 2 of 4 brands:
- **MetLife:** Returned "MetLife MDRT Committee" (31 subscribers) — an agent training channel, not the official brand channel. MetLife Korea effectively has no public YouTube presence.
- **Hanwha Life:** Returned "Hanwha Life Esports" — the e-sports team, not the insurance brand channel.

**Resolution:** Manual verification of all 4 channel IDs. MetLife flagged as `youtube_available: False` in config, automatically skipped by the YouTube collector. This asymmetry (3 brands with YouTube, 1 without) became a structural feature of the analysis, not a limitation to hide.

### Decision Point 2: Brand Name Unification

YouTube data used Korean names (삼성생명, 한화생명), while VoC data used English keys (samsung, hanwha). This caused brand splits in staging — the same brand appeared as two entries.

**Resolution:** Added `BRAND_KR_TO_KEY` mapping in the staging ETL, unified all downstream data to English keys. Detected during data quality check, fixed before full collection.

### Decision Point 3: Topic Granularity — 43 Topics → 6 Categories

BERTopic discovered 43 topics (+ outlier topic -1). Options considered:
- **Option A:** Re-run with `--nr-topics 15` to force fewer topics
- **Option B:** Keep 43 and manually map to business-meaningful categories

**Decision: Option B.** Rationale: reducing topic count risks losing meaningful granular topics (e.g., health campaigns, retirement content). Manual mapping preserves discovery while enabling business-level analysis.

Final mapping to 6 categories aligned with the project's content type framework:

| Category | Topics Mapped | Documents |
|---|---|---|
| 상품홍보 (Product) | 0, 6, 21, 24, 36, 40 | 3,334 |
| 교육·전문성 (Education) | 4, 5, 9, 10, 14, 15, 16, 23, 37, 41 | 1,530 |
| 고객후기 (Reviews) | 1, 17, 28, 33, 38, 42 | 1,519 |
| 건강·라이프 (Health) | 2, 3, 12, 13, 19, 20, 31, 39 | 1,493 |
| 이벤트·프로모션 (Events) | 8, 11, 18, 26, 27, 30, 34, 35 | 657 |
| noise | -1, 7, 22, 25, 29, 32 | 3,272 |

### Decision Point 4: Noise Re-Inspection

Initial noise classification assigned 3,582 docs. A targeted inspection (`scripts/inspect_noise.py`) revealed 6 topics (310 docs) that were actually meaningful:

| Rescued Topic | Docs | Reclassified To | Why |
|---|---|---|---|
| Topic 12 | 112 | 건강·라이프 | Hanwha "봉인하고 싶어요" health campaign |
| Topic 39 | 18 | 건강·라이프 | Same campaign (dementia focus) |
| Topic 16 | 96 | 교육·전문성 | Samsung Life AI/music content response |
| Topic 37 | 21 | 교육·전문성 | Kyobo Life "책을읽장" reading campaign |
| Topic 30 | 37 | 이벤트·프로모션 | Samsung Life sports sponsorship |
| Topic 34 | 26 | 이벤트·프로모션 | Samsung Life brand song campaign |

Confirmed noise (3,272 docs): BERTopic outliers (-1), emoticons (Topic 7), English/Spanish comments (22), Kyobo Bookstore confusion (25, 32), stock news listings (29).

### Decision Point 5: Topic 7 — Sentiment Yes, Topic No

Topic 7 (218 docs, short emotional reactions like "ㅋㅋㅋ", "유치하다") required special handling. Before deciding, the comments were traced back to their source videos:

**Finding:** 218 comments were distributed across **75 different videos** spanning sports, health, entertainment, and brand campaigns. No single campaign concentration.

**Decision:**
- **Topic classification: noise** — content type cannot be determined from the comment alone
- **Sentiment analysis: included** — emotional reactions are valid sentiment signals; the source video's topic category can be retrieved via video_id when needed

### Decision Point 6: Hybrid Sentiment Architecture

The 8,751 sentiment-eligible documents were processed in layers:

| Layer | Scope | Method | Result |
|---|---|---|---|
| Dictionary (high conf ≥ 0.3) | 2,640 docs (30.2%) | Insurance domain lexicon | Free, transparent |
| Dictionary (zero conf = 0) | 3,251 docs (37.2%) | No sentiment words detected | Classified as neutral |
| Claude Haiku (0 < conf < 0.3) | 2,860 docs (32.7%) | LLM API batch call | ~$0.1 total cost |

Key design: docs with confidence = 0 (no sentiment words at all) were **not** sent to Haiku. They are genuinely neutral — sending them to the API would waste cost without changing the label.

### Decision Point 7: Report VoC Window

The first report attempt showed MetLife VoC at only 6 documents for a single week — insufficient for meaningful sentiment ratios.

**Resolution:** Expanded the VoC query from single-week to **4-week rolling window**. This increased MetLife VoC to 96 documents, enabling directional sentiment analysis while clearly noting the sample limitation.

### Decision Point 8: Signal Matrix JOIN Bug

Initial signal matrix showed engagement = 0.0 for all cells. Root cause: `staging.content_enriched` stores YouTube comment `source_id` as the **comment_id**, but the query JOINed directly to `raw.youtube_videos` using video_id. The fix required routing through `raw.youtube_comments` first (comment → video → engagement stats).

---

## Results

### Sentiment Distribution (8,751 docs)

| Sentiment | Count | Ratio |
|---|---|---|
| Positive | 3,299 | 37.7% |
| Neutral | 4,946 | 56.5% |
| Negative | 505 | 5.8% |

### MetLife VoC Profile (4-week rolling)

| Metric | Value |
|---|---|
| Positive ratio | 24.5% |
| Negative ratio | 8.8% |
| Total docs | 96 |

### Lead-Lag Analysis

| Brand | Best Lag | Correlation | Interpretation |
|---|---|---|---|
| MetLife | +2 weeks | r = 0.22 | Content → search interest (weak positive) |
| Samsung | +8 weeks | r = 0.24 | Too distant — likely noise |
| Hanwha | -7 weeks | r = 0.24 | Search leads content — reactive pattern |
| Kyobo | +2 weeks | r = -0.22 | Content → search decrease (inverse) |

All correlations |r| < 0.3 — consistent with the project's framing as **pattern exploration, not causal confirmation**.

### Anomaly Detection (19 events across 4 brands)

Notable: 한화생명 2026-02-23 (ratio=100.0, z=7.49, 3/3 methods agreed) — the single largest anomaly in the dataset.

---

## Dashboard

**Streamlit 4-tab dashboard** with CSV fallback for cloud deployment.

| Tab | Content |
|---|---|
| Brand Tracker | SoV trend, WoW heatmap, pie chart |
| Content Performance | Signal matrix scatter, lead-lag bar chart |
| VoC Monitor | Sentiment stacked area, brand comparison, topic × sentiment |
| Weekly Report | AI report preview + PDF download |

---

## Project Structure

```
cx_reporting/
|-- collectors/
|   |-- naver_datalab.py        # Naver DataLab search trends
|   |-- youtube.py              # YouTube videos + comments (3 brands)
|   |-- naver_voc.py            # Blog/Cafe VoC (4 brands)
|   |-- app_reviews.py          # App Store / Google Play reviews
|-- etl/
|   |-- staging/
|   |   |-- text_preprocessing.py   # Text cleaning, brand unification
|   |-- mart/
|       |-- brand_search_weekly.py  # WoW, SoV calculation
|       |-- content_performance.py  # YouTube engagement metrics
|       |-- voc_sentiment.py        # Weekly sentiment aggregation
|-- analysis/
|   |-- topic_modeling.py       # BERTopic x LDA ensemble
|   |-- topic_mapping.py        # 43 topics -> 6 categories
|   |-- sentiment.py            # Hybrid dictionary + Claude Haiku
|   |-- signal_matrix.py        # Engagement x Search matrix + lead-lag
|   |-- anomaly_detection.py    # 3-way: Z-score + MSTL + IF
|-- report/
|   |-- weekly_report.py        # LLM narrative + PDF generation
|-- dashboard/
|   |-- app.py                  # Streamlit main (4 tabs)
|   |-- tabs/
|       |-- tab_brand.py        # Brand Tracker
|       |-- tab_content.py      # Content Performance
|       |-- tab_voc.py          # VoC Monitor
|       |-- tab_report.py       # Weekly Report
|-- scripts/
|   |-- run_pre_eda.py          # Pre-EDA validation runner
|   |-- eda_naver_blog_volume.py
|   |-- eda_youtube_channels.py
|   |-- eda_naver_datalab.py
|   |-- check_data_quality.py   # Phase 1 data quality audit
|   |-- inspect_noise.py        # Noise topic inspection
|   |-- collect_foreign_voc.py  # Foreign insurer community VoC
|   |-- process_app_reviews.py  # App review staging + sentiment
|   |-- init_schema.sql         # PostgreSQL 3-tier schema (11 tables)
|-- config/
|   |-- db.py                   # get_conn() context manager
|   |-- settings.py             # Brand keywords, API configs, thresholds
|-- data/
|   |-- raw/                    # API responses, pre-EDA results
|   |-- processed/              # Staging outputs (enriched text, sentiment)
|   |-- exports/                # Mart CSVs, PDF reports
|-- models/                     # Saved BERTopic, LDA models
|-- docker-compose.yml          # PostgreSQL 16
```

---

## How to Run

```bash
# 1. Environment setup
cp .env.example .env
# Fill in: NAVER_CLIENT_ID/SECRET, YOUTUBE_API_KEY, ANTHROPIC_API_KEY

# 2. Start PostgreSQL
docker-compose up -d

# 3. Initialize schema
psql -h localhost -p 5435 -U cx_admin -d cx_monitor -f scripts/init_schema.sql

# 4. Pre-EDA validation
python scripts/run_pre_eda.py

# 5. Data collection
python -m collectors.naver_datalab
python -m collectors.youtube --max-videos 100
python -m collectors.naver_voc --max-per-query 500
python -m collectors.app_reviews --count 200
python scripts/collect_foreign_voc.py --max-per-query 100

# 6. ETL
python -m etl.mart.brand_search_weekly
python -m etl.mart.content_performance
python -m etl.staging.text_preprocessing

# 7. NLP analysis
python -m analysis.topic_modeling
python -m analysis.topic_mapping
python -m analysis.sentiment
python scripts/process_app_reviews.py
# Note: BERTopic/LDA models (models/) are excluded from repo due to size (547MB).
# They are regenerated by running topic_modeling.py above.

# 8. Signal analysis
python -m etl.mart.voc_sentiment
python -m analysis.signal_matrix
python -m analysis.anomaly_detection

# 9. Report generation
python -m report.weekly_report

# 10. Dashboard
streamlit run dashboard/app.py
```

---

## Tech Stack

Python 3.10 · PostgreSQL 16 (Docker) · BERTopic · ko-sroberta-multitask · kiwipiepy · gensim (LDA) · Claude Haiku 4.5 (sentiment) · Claude Sonnet 4.6 (report) · MSTL · Isolation Forest · Streamlit · Plotly · ReportLab · Naver DataLab API · YouTube Data API v3

---

## Methodology Transfer from Prior Projects

| Prior Project | Transferred Element | Application |
|---|---|---|
| sportswear-brand-monitor | PostgreSQL 3-tier, CSV fallback, anomaly 3-way | Schema design, Streamlit Cloud deploy, anomaly detection |
| cnp-voc-pipeline | BERTopic × LDA, YouTube/Blog collectors | Topic modeling ensemble, data collection |
| sportswear-brand-monitor | Hybrid sentiment (dictionary + LLM) | Insurance domain lexicon + Claude Haiku routing |
| sportswear-brand-monitor | MSTL lead-lag analysis | Content → search interest cross-correlation |
| pg-hns-consumer-signal-pipeline | Naver DataLab collector | Brand search trend collection |

---