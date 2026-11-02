# 손해보험 VoC·브랜드 시그널 분석 자동화 — AI Weekly CX Report

> **Target:** KB손해보험 (vs. 삼성화재 · 현대해상 · DB손해보험 · 메리츠화재)
> **Positioning:** 공개 데이터 기반 VoC 분석 파이프라인 + 통계 검정 + AI 주간 리포트 자동화 — 데이터 기반 의사결정 지원 PoC

**Independent Project** · Live dashboard: https://cx-reporting-ai-report.streamlit.app/

---

## Overview

손해보험 5사의 공개 VoC(블로그·카페·YouTube 댓글), 네이버 검색 관심도, 공식 YouTube 콘텐츠를 수집해 **계층적 토픽 모델링 → 하이브리드 감성 분석 → 브랜드×카테고리 통계 검정 → AI 주간 리포트**로 이어지는 파이프라인을 구축했다. 모든 수치에는 시그널 강도 태그(`통계 검증 ✓` / `cluster 시그널` / `정성 시그널`)를 붙여 리포트 독자가 단정 가능한 수치와 참고용 패턴을 구분할 수 있게 했다.

> **경계:** 인과 관계를 확정하지 않는다. 브랜드 간 비교는 대칭 수집 서브셋에서만 검정하고, 타겟 브랜드 심층 VoC는 별도 수집분으로 리포트와 대표 문서 추출에만 쓴다.

---

## Key Findings

1. **KB손해보험 VoC는 상품·보상 중심, 경쟁사는 캠페인 반응 중심** `통계 검증 ✓` — 대칭 수집 서브셋 기준 KB손해보험은 보험금 청구·보상 VoC 비중이 경쟁 4사 평균 대비 **5.10×**, 자동차보험 **2.88×**, 실손·건강보험 **1.79×** 과대 인덱싱(Bonferroni α=0.05/30). 반면 현대해상(브랜드 캠페인 1.64×)·DB손해보험(1.28×)은 YouTube 캠페인 댓글이 VoC를 지배한다. KB손해보험 고객은 실제 서비스 접점 경험을 더 많이 공유하며, 그만큼 보상·청구 단계 감성에 노출돼 있다.

2. **빅4 접전 구도** — 검색 SoV 기준 삼성화재 22.6% · 현대해상 22.8% · KB손해보험 21.2% · DB손해보험 20.5%로 1~2%p 내 경합. 2026-W44 기준 KB손해보험이 빅4 중 유일한 WoW 상승(+1.7%).

3. **업계 동반 하락 = 연휴 효과** `통계 검증 ✓` — 3-way 이상탐지가 5사 **동시** 급락 주 4개를 잡아냈고(2024-09-16, 2025-01-27, 2025-10-06, 2026-09-21), 전부 추석·설 연휴 주였다. 단일 브랜드 모니터링으로는 "우리만 떨어졌다"로 오독할 수 있는 패턴을 5사 교차 분석이 자동으로 걸러낸다.

4. **UBI(안전운전 할인) 상품이 자발적 긍정 VoC를 만든다** `정성 시그널` — KB손해보험 자동차보험 카테고리 긍정 대표 문서가 "KB다이렉트 운전점수 95점, 걸음수 5000보, 보험료 환급" 같은 1인칭 참여 후기로 나타났다. 프로그램 참여 자체가 콘텐츠가 되는 구조.

5. **부정 VoC의 핵심은 손해액 산정 불신** `정성 시그널` — "상대방 보험사에서 100대 0… 전손은 안 되겠죠", "손해액 산정 기준은" 등 보상 절차 불확실성이 자동차보험 부정 담론의 중심.

6. **감성 비율은 안정, 밀도가 핵심** `cluster 시그널` — KB손해보험 부정 8.1%(5주 롤링, n=1,134)는 낮지만, 그 부정이 청구·보상 카테고리에 집중돼 있어 비율만 보면 과소평가된다.

---

## Pipeline

```
[Collection — 3 collectors, settings-driven]
|-- Naver DataLab API: 5사 주간 검색 관심도 (148주)
|-- YouTube Data API v3: 5사 공식 채널 영상 458 + 댓글 6,329
|-- Naver Search API: Blog/Cafe VoC 7,603 (브랜드 후기 + KB 상품별 심층 쿼리)
         |
         v
[ETL — PostgreSQL 3-tier]  raw → staging → mart
|-- 브랜드명 통일 (settings.BRAND_KR_TO_KEY), 텍스트 정제 → staging 13,595건
         |
         v
[NLP — 3-stage hierarchical topic modeling]
|-- Stage 1  BERTopic(ko-sroberta) × LDA: 59 topics, 브랜드명·퀴즈 토큰 불용어
|-- Stage 2  Topic 0 (상품 VoC 4,455건) 재클러스터링 → 10 subtopics (100번대)
|-- Stage 2' Topic 100 (자동차·실손 혼합 2,297건) 재클러스터링 → 15 subtopics (200번대)
|-- Stage 3  Outlier 재배정: 임베딩 centroid 유사도 ≥ 중앙값(0.65) → 53% 복구
|-- 수동 매핑 → 7 카테고리 (noise 23.2%)
|-- 하이브리드 감성: 손해보험 사전(confidence ≥ 0.3) → Claude Haiku(0 < conf < 0.3)
         |
         v
[Signal Analysis]
|-- 브랜드×카테고리 카이제곱 + Bonferroni post-hoc z-test (대칭 서브셋)
|-- Engagement × 검색 관심도 매트릭스
|-- Lead-lag 교차상관 (MSTL 잔차)
|-- 3-way 이상탐지 (Z-score + MSTL + Isolation Forest, 2/3 합의)
|-- 대표 문서 추출 (소스 우선순위 + 광고 마커 제외)
         |
         v
[AI Report — Claude Sonnet → ReportLab PDF]
|-- 6섹션 1페이지, 전 수치 시그널 태그, 원인 후보 분리, 원문 인용
         |
         v
[Dashboard — Streamlit 4 tabs, CSV fallback]
```

---

## Business Context

KB손해보험은 KB금융그룹 계열 손해보험사로 자동차보험·장기(실손·건강)·일반보험을 다이렉트와 설계사 채널 양쪽으로 판매한다. 검색 관심도 기준 삼성화재·현대해상·DB손해보험과 20% 초반대 SoV로 경합하는 빅4 구도이며, 메리츠화재가 그 아래에 있다.

손해보험 CX의 핵심 접점은 **보험금 청구·보상 처리 경험**과 **자동차보험 갱신 시점**이다. 이 프로젝트의 카테고리 프레임(자동차보험 / 실손·건강보험 / 보험금 청구·보상 / 상품 설계·고객후기 / 브랜드 캠페인 / 이벤트·프로모션)은 이 접점 구조를 반영해 설계했다.

---

## Data Sources

| Data | Source | Volume | 비고 |
|---|---|---|---|
| 브랜드 검색 관심도 | Naver DataLab API | 740 rows (5사 × 148주) | 상대값, 5그룹 상한 |
| 공식 YouTube 영상·댓글 | YouTube Data API v3 | 458 영상, 6,329 댓글 | 메리츠화재는 채널 전체 59개 |
| 소비자 VoC | Naver Search API (Blog/Cafe) | 7,603 unique | Cafe는 postdate 미제공 |

### Pre-EDA (2026-11-02)

| Check | Result |
|---|---|
| Blog VoC 월간 볼륨 | 5사 모두 ≥ 200건 |
| YouTube 공식 채널 | 5사 모두 확인 (현대해상은 검색어 변경 후 발견, 메리츠화재 6.7K 소규모) |
| DataLab SoV | 삼성화재 22.9 / KB손해보험 22.0 / 현대해상 21.7 / DB손해보험 20.5 / 메리츠화재 12.8 |

---

## Methodology & Decision Points

### DP1. 단일 BERTopic이 장르 혼합 데이터에서 실패 → 계층적 모델링

Stage 1에서 Topic 0이 4,455건(33%)으로 블랙홀이 됐다. 블로그/카페 상품 글은 임베딩 공간에서 한 덩어리로 뭉치고, YouTube 댓글만 잘게 쪼개진 것이다. LDA 교차검증 매칭률 30%가 내부 이질성을 확인해줬다. 불용어 보강(브랜드명·퀴즈 토큰)으로는 풀리지 않아 **Topic 0 → Stage 2 재클러스터링(10 subtopics) → 그중 Topic 100(자동차+실손 혼합) → Stage 2' 재클러스터링(15 subtopics)** 3단계로 갔다. 실손보험 언급 467건이 자동차보험 토픽에 묻혀 있던 것이 이 단계에서 분리됐다.

### DP2. Outlier 재배정 임계값 = 중앙값

HDBSCAN outlier 2,890건(21%) 중 상당수가 실제 상품 VoC였다. 임베딩 centroid 코사인 유사도로 재배정하되, 첫 시도(threshold 0.55, 80% 재배정)는 p25(0.572)보다 낮아 "noise를 줄이려고 임계값을 맞췄다"는 비판을 피할 수 없었다. **중앙값(0.65)으로 올려 53%만 재배정** — 한 문장으로 설명 가능한 규칙을 택했다. 최종 noise 23.2%.

### DP3. Noise 재검토 → 브랜드 캠페인 1,331건 복구

초기 noise 43%를 토픽별로 점검하니, KB손해보험 "일상" 캠페인(라무·비비), DB손해보험 "수고했다 나 자신", 삼성화재 안내견 CSR, 현대해상 이름·찐친·육아 시리즈 등 **브랜드 캠페인 댓글 1,331건**이 noise에 있었다. 보험 상품 얘기는 아니지만 브랜드 콘텐츠 engagement이므로 "브랜드 캠페인" 카테고리로 복구했다. 확정 noise에는 "해상도"(TV·오디오)가 현대해상으로 오태깅된 ~280건, 변호사·중고차·부동산 광고, 채용 공고가 남았다.

### DP4. 감성사전에서 "손해" 제거

손해보험 도메인에서 "손해"는 "손해보험·손해배상·손해사정"처럼 중립 용어인데 -0.8로 들어가 있었다. 첫 실행에서 고신뢰 negative 876건 중 **726건(83%)이 "손해" 매칭**이었다. 제거 후 negative 드라이버가 피해·해지·거절·인상·분쟁으로 바뀌었다 — 실제 불만 어휘다.

### DP5. 브랜드×카테고리 검정의 수집 편향 → 대칭 서브셋

첫 검정에서 KB손해보험 보험금 청구 9.06×, 자동차보험 6.57×가 나왔지만, 이는 KB손해보험에만 상품별 쿼리 4개("KB손해보험 자동차보험/실손보험/보험금 청구/다이렉트")가 있었기 때문이었다. Bonferroni 검정이 수집 편향을 "유의"로 확인해준 셈이다. **타겟 브랜드 상품별 쿼리 문서를 교차 비교에서 제외**하자 5.10× / 2.88×로 내려갔지만 여전히 유의했다 — 이제 진짜 시그널이다. 타겟 심층 VoC는 리포트·대표 문서에만 쓴다.

### DP6. 대표 문서 = 감성 강도 최대가 아니다

|score| 최대 순으로 뽑으니 "음주뺑소니 변호사", "손해사정사 무료상담" 같은 법률 광고가 부정 대표로 올라왔다. **소스 우선순위(YouTube 댓글 → 카페 → 블로그) + 광고 마커 정규식 제외**를 적용하자 "상대방 보험사 100대 0… 전손은 안 되겠죠" 같은 1인칭 경험담이 올라왔다. 토픽 집계에는 영향 없는 인용 적합성 필터다.

### DP7. 시그널 강도 태그 체계 (방법론 전이)

이전 자사몰 리뷰 분석(19K건)에서 쓴 **통계 검증 ✓ / cluster 시그널 / 정성 시그널** 태그를 가져왔다. 리포트의 모든 수치에 태그를 붙이고, `통계 검증 ✓`만 단정 표현을 허용한다. 공개 웹 데이터는 1차 데이터보다 noise가 많으므로 이 단서 체계가 오히려 더 필요하다. 같은 분석에서 가져온 Bonferroni 다중검정과 대표 문서 추출도 여기 적용했고, 시즌 캘린더·20개 액션 카탈로그는 주간 리포트 포맷에 맞지 않아 가져오지 않았다.

### DP8. 포트 충돌·모델 로드 경로

PostgreSQL 포트 5435가 다른 프로젝트 컨테이너에 선점돼 5438로 이전. BERTopic stage-2 모델 저장 경로가 첫 실행과 두 번째 실행에서 달라 `BERTopic.load`가 HuggingFace Hub로 빠지는 오류 — 로컬 경로 통일로 해결. 재현성을 위해 README에 기록.

---

## Results

### 카테고리 분포 (13,595건)

| 카테고리 | 건수 | 비율 |
|---|---|---|
| 브랜드 캠페인 | 3,934 | 28.9% |
| 상품 설계·고객후기 | 2,371 | 17.4% |
| 이벤트·프로모션 | 1,790 | 13.2% |
| 자동차보험 | 1,251 | 9.2% |
| 실손·건강보험 | 878 | 6.5% |
| 보험금 청구·보상 | 212 | 1.6% |
| noise | 3,159 | 23.2% |

### 감성 (10,436건, noise 제외)

| | Positive | Neutral | Negative |
|---|---|---|---|
| 전체 | 45.7% | 47.8% | 6.5% |
| 처리 방식 | 사전 4,226 · Haiku 3,153 · 중립 확정 3,057 |

### 브랜드×카테고리 유의 인덱싱 (대칭 서브셋, Bonferroni α=0.05/30)

| 브랜드 | 과대 | 과소 |
|---|---|---|
| **KB손해보험** | 보험금 청구·보상 5.10× · 자동차보험 2.88× · 실손 1.79× | 브랜드 캠페인 0.69× |
| 현대해상 | 브랜드 캠페인 1.64× | 자동차 0.11× · 청구 0.11× · 실손 0.19× |
| DB손해보험 | 브랜드 캠페인 1.28× · 이벤트 1.30× | 상품후기 0.41× · 실손 0.29× |
| 삼성화재 | 이벤트 1.68× · 상품후기 1.80× · 자동차 2.25× | 브랜드 캠페인 0.33× |
| 메리츠화재 | 실손 7.43× · 상품후기 3.92× | 브랜드 캠페인 0.09× |

### 이상탐지 (26 events, 2/3 methods)

5사 동시 급락 주: 2024-09-16 · 2025-01-27 · 2025-10-06 · 2026-09-21 (추석·설). 복수 브랜드 동시 급등: 2025-12-08 · 2026-02-23.

### Lead-lag

KB손해보험 lag=+6 r=−0.52, DB손해보험 lag=+3 r=−0.54 — VoC 볼륨 기반이라 광고 물량 교란 가능성이 있어 `정성 시그널`로 해석 보류.

---

## Project Structure

```
cx_reporting/
|-- config/
|   |-- settings.py              # 5사 브랜드·키워드·채널 ID·TARGET_BRAND (단일 진실원)
|   |-- db.py
|-- collectors/
|   |-- naver_datalab.py
|   |-- youtube.py
|   |-- naver_voc.py             # BRAND_PATTERNS를 settings에서 생성
|-- etl/
|   |-- staging/text_preprocessing.py
|   |-- mart/{brand_search_weekly, content_performance, voc_sentiment}.py
|-- analysis/
|   |-- topic_modeling.py        # Stage 1 BERTopic × LDA
|   |-- topic_modeling_stage2.py # 계층 재클러스터링 (--parent-topic, --offset)
|   |-- reduce_outliers.py       # Stage 3 centroid 유사도 재배정
|   |-- topic_mapping.py         # 7 카테고리 수동 매핑
|   |-- sentiment.py             # 손해보험 사전 + Claude Haiku
|   |-- signal_matrix.py         # 매트릭스·Bonferroni 검정·lead-lag·대표 문서
|   |-- anomaly_detection.py
|-- report/weekly_report.py      # Claude Sonnet → PDF, 시그널 태그
|-- dashboard/app.py, tabs/
|-- scripts/
|   |-- run_pre_eda.py
|   |-- check_data_quality.py
|   |-- inspect_noise.py
|   |-- init_schema.sql
|-- data/exports/                # CSV (Streamlit Cloud fallback) + PDF
|-- models/                      # (gitignored, 547MB) — topic_modeling.py로 재생성
```

---

## How to Run

```bash
cp .env.example .env            # NAVER / YOUTUBE / ANTHROPIC 키
docker-compose up -d            # PostgreSQL :5438
psql -h localhost -p 5438 -U cx_admin -d cx_monitor -f scripts/init_schema.sql

python scripts/run_pre_eda.py   # 채널 ID 확인 → settings.py 반영

python -m collectors.naver_datalab
python -m collectors.youtube --max-videos 100
python -m collectors.naver_voc --max-per-query 500

python -m etl.mart.brand_search_weekly
python -m etl.mart.content_performance
python -m etl.staging.text_preprocessing

python -m analysis.topic_modeling
python -m analysis.topic_modeling_stage2 --parent-topic 0   --min-topic-size 40 --offset 100
python -m analysis.topic_modeling_stage2 --parent-topic 100 --min-topic-size 30 --offset 200
python -m analysis.reduce_outliers --threshold 0.65
python -m analysis.topic_mapping
python -m analysis.sentiment

python -m etl.mart.voc_sentiment
python -m analysis.signal_matrix
python -m analysis.anomaly_detection
python -m report.weekly_report

streamlit run dashboard/app.py
```

---

## Tech Stack

Python 3.10 · PostgreSQL 16 · BERTopic · ko-sroberta-multitask · kiwipiepy · gensim · scipy · statsmodels (MSTL) · scikit-learn (Isolation Forest) · Claude Haiku 4.5 / Sonnet 4.6 · Streamlit · Plotly · ReportLab · Naver DataLab API · YouTube Data API v3

---

*Independent Project · GitHub: [ChangyeolAidenOh](https://github.com/ChangyeolAidenOh)*
