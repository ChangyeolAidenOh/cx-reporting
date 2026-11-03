# 손해보험 VoC·브랜드 시그널 분석 자동화 — AI Weekly CX Report

> **Target:** KB손해보험 (vs. 삼성화재 · 현대해상 · DB손해보험 · 메리츠화재)
> **Positioning:** 공개 데이터 기반 VoC 분석 파이프라인 + 소스·작성자 층화 통계 검정 + 주간 자동화 + AI 주간 리포트 — 데이터 기반 의사결정 지원 PoC

**Independent Project** · Live dashboard: https://cx-reporting-ai-report.streamlit.app/

---

## Overview

손해보험 5사의 공개 VoC(블로그·카페·YouTube 댓글), 네이버 검색 관심도, 공식 YouTube 콘텐츠를 수집해 **계층적 토픽 모델링 → 서빙 분류기 증류 → 작성자 축 라벨링 → 하이브리드 감성 분석 → 소스×작성자 층화 브랜드×카테고리 검정 → 주간 증분 파이프라인 → AI 주간 리포트**로 이어지는 시스템을 구축했다. 모든 수치에는 세그먼트와 시그널 강도 태그(`통계 검증 ✓` / `cluster 시그널` / `정성 시그널` / `표본 부족`)를 붙여, 리포트 독자가 단정 가능한 수치와 참고용 패턴을 구분할 수 있게 했다.

> **경계:** 인과 관계를 확정하지 않는다. 브랜드 간 비교는 소스·작성자로 층화한 세그먼트의 대칭 수집 서브셋에서만 검정하고, `통계 검증 ✓`는 Bonferroni 유의와 leave-one-out 강건성을 모두 통과한 셀에만 붙인다. 타겟 브랜드 심층 VoC는 별도 수집분으로 리포트와 대표 문서 추출에만 쓴다.

이 프로젝트의 가장 중요한 결과는 **처음 내놓은 헤드라인을 스스로 기각한 것**이다. "KB손해보험 VoC는 보험금 청구·보상 5.10× 과대 인덱싱"은 수집 쿼리 비대칭 → 설계사 콘텐츠 → 소스 믹스 → 택소노미 누락 → one-vs-rest 거울상, 다섯 교란을 차례로 걷어내는 동안 사라졌고, 그 자리에 남은 것은 "고객 VoC 구성에서 KB손해보험은 경쟁사와 구분되지 않는다(검출 한계 내)"는 더 정직한 발견과, 그 발견이 어디까지 말할 수 있는지를 숫자로 적는 검정 규칙이다.

---

## Key Findings

모든 수치는 세그먼트와 시그널 강도 태그를 함께 표기한다. `통계 검증 ✓`는 Bonferroni 유의 **그리고** leave-one-out(경쟁사를 하나씩 빼도 유의) 조건을 모두 통과한 셀에만 붙인다. 판정 규칙은 [docs/brand_category_test_rules.md](docs/brand_category_test_rules.md).

1. **공개 웹 고객 VoC의 상품 구성에서 KB손해보험은 경쟁사와 구분되지 않는다** `통계 검증 ✓ (검출 한계 명시)` — blog·cafe 고객 작성 글(`web_customer`, n=900, KB손해보험 131)에서 KB손해보험의 robust 유의 셀은 0개. 상품 설계·고객후기 1.38×는 검출 범위(0.51–1.49×) 안이고, 실손·건강보험 1.33×(검출 범위 0.26–1.74×)와 자동차보험 1.53×(n=15, 2.28× 이상만 검출 가능, 수축 추정 1.46× [0.87, 2.18])은 표본 부족으로 **미확정**이다. 이전 헤드라인 "보험금 청구·보상 5.10×, 자동차 2.88× 과대 인덱싱"은 설계사 콘텐츠·소스 믹스·수집 쿼리 비대칭의 산물로 기각했다.

2. **현대해상 고객 VoC는 태아·어린이보험에 집중된다** `통계 검증 ✓` — `web_customer`에서 현대해상 글의 78.1%가 태아·어린이보험(5.57×, n=217, LOO 5.2–6.2×). 5사 중 유일하게 robust한 브랜드 차별 요인이다. 현대해상의 상품 설계·실손 과소(0.30×·0.24×)와 나머지 4사의 태아 과소(0.30–0.46×, 전부 fragile)는 이 집중의 거울상이지 각 사의 특성이 아니다. 이 카테고리는 원래 "상품 설계·고객후기"에 묻혀 있었고(고객 글의 41%), 분리 전에는 현대해상 상품설계 1.47×라는 틀린 발견을 만들었다.

3. **설계사·비교 콘텐츠는 KB손해보험을 자동차보험(다이렉트) 중심으로 다룬다** `통계 검증 ✓` — `web_advisor`(n=1,001, KB손해보험 311) 자동차보험 1.88×(LOO 1.48–2.70×), 태아·어린이보험 0.32×(LOO 0.27–0.57×). 공급자 측 신호다. 비교·가격 콘텐츠의 주요 소재라는 뜻이지 고객 불만의 근거가 아니며, 고객 세그먼트에서는 같은 셀이 미확정이다.

4. **공개 웹 보험 VoC에서 "고객 목소리"는 소수다** `cluster 시그널` — blog·cafe 8,683건의 작성자 구성은 customer 17%, advisor 30%, media 2%, unrelated 51%. 고객이 직접 쓴 보험금 청구·보상 글은 5사 통틀어 15건이라 blog·cafe는 청구 경험 VoC 채널이 아니다. 작성자 축을 걷어내기 전의 "청구·보상 VoC"는 손해사정사·설계사 콘텐츠였다.

5. **5사 동시 검색 급락 = 명절** `통계 검증 ✓` — 3-way 이상탐지가 잡은 5사 동시 급락 주 4개(2024-09-16, 2025-01-27, 2025-10-06, 2026-09-21)는 전부 추석·설 연휴 주였다. 단일 브랜드 모니터링은 이를 브랜드 이슈로 오독한다.

6. **YouTube 반응 구성은 채널 운영 방식을 반영한다** `통계 검증 ✓` — `youtube` 세그먼트(n=5,608)에서 삼성화재 이벤트·프로모션 2.51×, 현대해상 브랜드 캠페인 1.30×, DB손해보험 상품 설계·고객후기 6.30×(전부 robust). KB손해보험은 브랜드 캠페인 0.97×(검출 범위 0.91–1.09×)로 경쟁사와 동일. 소스를 합친 검정에서는 YouTube 구독자 규모(DB손해보험 59만 · 삼성화재 49만 vs KB손해보험 13.5만 · 메리츠화재 6.7천)가 VoC 구성을 지배하므로, 풀링 수치는 참고용으로만 둔다.

검출 한계와 표본 목표: `web_customer`에서 KB손해보험 자동차보험 셀이 2× 차이를 검출하려면 브랜드 문서 약 220건, 1.5×는 약 860건이 필요하다(현재 131). 주간 수집 누적과 경쟁사 대칭 상품 쿼리 확장이 이 한계를 좁히는 경로다.

---

## Pipeline

```
[Collection — 3 collectors, settings-driven]
|-- Naver DataLab API: 5사 주간 검색 관심도 (148주, 직전 주 소급 수정 → settle 3일 후 확정)
|-- YouTube Data API v3: 5사 공식 채널 영상 458 + 댓글 6,329
|-- Naver Search API: Blog/Cafe VoC 7,603 (브랜드 후기 5 + 일반 2 + 타겟 상품 쿼리 4)
         |
         v
[ETL — PostgreSQL 3-tier]  raw → staging → mart (운영 DB Supabase, 개발 DB Docker)
|-- 브랜드명 통일, 텍스트 정제 → staging 14,899건 (초기 13,595 + 주간 증분)
|-- 브랜드 귀속: 제목에 다른 브랜드 하나만 언급되면 그 브랜드로 (brand_source='title')
         |
         v
[NLP — 3-stage hierarchical topic modeling → serve classifier]
|-- Stage 1  BERTopic(ko-sroberta) × LDA: 59 topics
|-- Stage 2  Topic 0 → 10 subtopics (100번대), Topic 100 → 15 subtopics (200번대)
|-- Stage 3  Outlier 재배정: centroid 유사도 ≥ 중앙값(0.65) → 53% 복구
|-- 수동 매핑 7 카테고리 + 규칙 분리 1 (태아·어린이보험) = 8 카테고리
|-- 서빙: kNN 분류기(ko-sroberta 임베딩, 8 클래스) CV 95.6%, conf ≥ 0.7에서 99.2% / coverage 89%
|-- 작성자 축 author_type: youtube 댓글은 규칙(customer), blog·cafe는 분류기(소스별 게이트) → 미달은 Haiku 2단계 → blog customer는 Sonnet 재검증
|-- 작성자 unrelated → noise (모든 하류 모듈에서 제외)
|-- 하이브리드 감성: 손해보험 사전(confidence ≥ 0.3) → Claude Haiku(0 < conf < 0.3)
         |
         v
[Signal Analysis — 소스 × 작성자 층화]
|-- 브랜드×카테고리 검정: χ² + Bonferroni post-hoc(n<30은 Fisher), 셀별 MDE, leave-one-out 강건성, 소표본 수축 추정
|   세그먼트 web_customer / web_advisor / youtube (+ all_pooled 참고용)
|-- Engagement × 검색 관심도 매트릭스
|-- Lead-lag 교차상관 (MSTL 잔차)
|-- 3-way 이상탐지 (Z-score + MSTL + Isolation Forest, 2/3 합의)
|-- 대표 문서 추출 (소스 우선순위 + 광고 마커 제외; 리포트 인용은 상품·여정 카테고리 고객 글만)
         |
         v
[Weekly automation — orchestration/]
|-- 증분 수집 → 제목 브랜드 규칙 → 카테고리 분류기 → 작성자 분류기 → LLM 검토 → 작성자 필터 → tie-break → 감성
|-- 마트 재생성 → 주간 스냅샷·전주 대비 변화 → export → AI 리포트
         |
         v
[AI Report — Claude Sonnet → ReportLab PDF]  /  [Dashboard — Streamlit, CSV fallback]
```

---

## Business Context

KB손해보험은 KB금융그룹 계열 손해보험사로 자동차보험·장기(실손·건강)·일반보험을 다이렉트와 설계사 채널 양쪽으로 판매한다. 검색 관심도 기준 삼성화재·현대해상·DB손해보험과 20% 초반대 SoV로 경합하는 빅4 구도이며, 메리츠화재가 그 아래에 있다.

카테고리 프레임은 상품(자동차보험 / 실손·건강보험 / 태아·어린이보험) · 여정(보험금 청구·보상) · 담론(상품 설계·고객후기) · 콘텐츠(브랜드 캠페인 / 이벤트·프로모션)의 네 축이다. 손해보험 VoC는 "무엇에 대한 글인가"만으로는 읽히지 않는다 — 공개 웹 글의 절반은 보험과 무관하고 3분의 1은 설계사·비교 콘텐츠이므로, 모든 분석은 "누가 썼는가"(작성자 축)와 "어느 채널인가"(소스 축)를 먼저 가른 뒤에 한다.

---

## Data Sources

| Data | Source | Volume | 비고 |
|---|---|---|---|
| 브랜드 검색 관심도 | Naver DataLab API | 740 rows (5사 × 148주) | 상대값, 5그룹 상한, 직전 주 소급 수정 |
| 공식 YouTube 영상·댓글 | YouTube Data API v3 | 458 영상, 6,329 댓글 | 메리츠화재는 채널 전체 59개 |
| 소비자 VoC | Naver Search API (Blog/Cafe) | 7,603 unique (+ 주간 증분) | Cafe는 postdate 미제공 |

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

HDBSCAN outlier 2,890건(21%) 중 상당수가 실제 상품 VoC였다. 임베딩 centroid 코사인 유사도로 재배정하되, 첫 시도(threshold 0.55, 80% 재배정)는 p25(0.572)보다 낮아 "noise를 줄이려고 임계값을 맞췄다"는 비판을 피할 수 없었다. **중앙값(0.65)으로 올려 53%만 재배정** — 한 문장으로 설명 가능한 규칙을 택했다.

### DP3. Noise 재검토 → 브랜드 캠페인 1,331건 복구

초기 noise 43%를 토픽별로 점검하니, KB손해보험 "일상" 캠페인(라무·비비), DB손해보험 "수고했다 나 자신", 삼성화재 안내견 CSR, 현대해상 이름·찐친·육아 시리즈 등 **브랜드 캠페인 댓글 1,331건**이 noise에 있었다. 보험 상품 얘기는 아니지만 브랜드 콘텐츠 engagement이므로 "브랜드 캠페인" 카테고리로 복구했다. 확정 noise에는 "해상도"(TV·오디오)가 현대해상으로 오태깅된 ~280건, 변호사·중고차·부동산 광고, 채용 공고가 남았다.

### DP4. 감성사전에서 "손해" 제거

손해보험 도메인에서 "손해"는 "손해보험·손해배상·손해사정"처럼 중립 용어인데 -0.8로 들어가 있었다. 첫 실행에서 고신뢰 negative 876건 중 **726건(83%)이 "손해" 매칭**이었다. 제거 후 negative 드라이버가 피해·해지·거절·인상·분쟁으로 바뀌었다 — 실제 불만 어휘다.

### DP5. 브랜드×카테고리 검정의 수집 편향 → 대칭 서브셋 → 그래도 부족했다

첫 검정에서 KB손해보험 보험금 청구 9.06×, 자동차보험 6.57×가 나왔지만, 이는 KB손해보험에만 상품별 쿼리 4개가 있었기 때문이었다. **타겟 브랜드 상품별 쿼리 문서를 교차 비교에서 제외**하자 5.10× / 2.88×로 내려갔고 여전히 유의했다 — 당시에는 이것을 진짜 시그널로 썼다. 그러나 DP10(작성자 축)과 DP13(소스 층화)을 거치자 이 셀은 설계사·손해사정사 콘텐츠와 YouTube 채널 규모 차이의 산물로 드러나 **기각**했다. 교란 하나를 걷어낸 뒤 남은 유의성이 곧 발견은 아니라는 것이 이 프로젝트의 핵심 교훈이다.

### DP6. 대표 문서 = 감성 강도 최대가 아니다

|score| 최대 순으로 뽑으니 "음주뺑소니 변호사", "손해사정사 무료상담" 같은 법률 광고가 부정 대표로 올라왔다. **소스 우선순위(YouTube 댓글 → 카페 → 블로그) + 광고 마커 정규식 제외**를 적용하자 1인칭 경험담이 올라왔다. 작성자 축 도입 후에는 리포트 인용을 상품·여정 카테고리의 고객 작성 글로 한정한다 — 캠페인 영상 댓글은 감성이 강해도 CX VoC가 아니다.

### DP7. 시그널 강도 태그 체계 (방법론 전이)

이전 자사몰 리뷰 분석(19K건)에서 쓴 **통계 검증 ✓ / cluster 시그널 / 정성 시그널** 태그를 가져왔다. 리포트의 모든 수치에 태그를 붙이고, `통계 검증 ✓`만 단정 표현을 허용한다. 공개 웹 데이터는 1차 데이터보다 noise가 많으므로 이 단서 체계가 오히려 더 필요하다. 이 프로젝트에서 `표본 부족`(미확정) 태그를 추가했다.

### DP8. 포트 충돌·모델 로드 경로

PostgreSQL 포트 5435가 다른 프로젝트 컨테이너에 선점돼 5438로 이전. BERTopic stage-2 모델 저장 경로가 첫 실행과 두 번째 실행에서 달라 `BERTopic.load`가 HuggingFace Hub로 빠지는 오류 — 로컬 경로 통일로 해결. 운영 DB는 이후 Supabase로 이관했다.

### DP9. 매주 재학습하면 토픽 ID가 바뀐다 → train/serve 분리와 분류기 증류

주간 자동화를 붙이자 BERTopic 재학습마다 토픽 ID가 바뀌어 수동 매핑이 무효가 되는 설계 결함이 드러났다. 첫 시도(safetensors로 모델 고정)는 원 토픽 재현율 87.8%에 그쳐 **기각**. 대신 최종 카테고리 라벨을 ko-sroberta 임베딩 위의 kNN 분류기로 증류했다. 측정 지표를 토픽 ID 일치율에서 **카테고리 정확도**로, 평균 정확도에서 **accuracy@coverage**(신뢰도 게이트 적용 시 자동 확정 비율과 그 정확도)로 바꾸자 설계가 명확해졌다: knn10, conf ≥ 0.7에서 자동 확정 89% / 정확도 99.2%, 나머지는 검토 큐 → Haiku tie-break. 학습 집합에서 분류기 자신의 주간 출력은 제외해 self-training을 막는다.

### DP10. "누가 썼는가"를 구분하지 않은 것이 세 문제의 공통 원인 → 작성자 축

tie-break로 구제한 문서를 샘플링하니 절반이 설계사·손해사정사 영업 블로그였다. 대표 문서 오염(DP6), lead-lag 교란, 청구·보상 과대 인덱싱(DP5)의 공통 원인이 "주제"만 분류하고 "작성자"를 분류하지 않은 것이었다. author_type(customer / advisor / media / unrelated) 축을 도입했다: YouTube 댓글은 규칙(customer), blog·cafe는 Haiku 2단계 프롬프트("보험이 주제인가 → 누가 썼는가", `insurance_topic` 구조화 출력을 코드에서 강제), customer 라벨은 Sonnet 재검증(26% 수정). 전체 LLM 대비 비용 60% 절감. 결과: blog·cafe의 50%가 보험 무관, 30%가 설계사 콘텐츠, 고객 글은 17%. 주간 서빙은 소스별 kNN 분류기(게이트 blog 0.8·0.85 / cafe 0.7, coverage 71%, customer precision 92%, unrelated precision 98%)가 맡고 게이트 미달만 LLM으로 보낸다.

### DP11. 택소노미에 없던 상품 축 → 태아·어린이보험 분리

작성자 층화 후 "현대해상 상품 설계·고객후기 1.47×"가 남았는데, 샘플 20건이 전부 태아보험 글이었다. 고객 작성 상품설계 글의 41%가 태아·어린이보험이었고(현대해상은 82%) 카테고리가 없어 담론 축에 묻혀 있었다. 정규식 규칙으로 분리(샘플 정밀도 38/40, 20/20)하고 8번째 카테고리로 추가했다. 분리 전 한 번 시도했던 "분류기 0.85 재라벨"은 태아보험 글이 자동차보험으로 가서 기각했었는데, 태아 클래스가 생기자 같은 재라벨이 안전해졌다 — 실패한 방법이 틀린 게 아니라 전제가 빠져 있었던 경우다.

### DP12. 브랜드 귀속이 수집 쿼리를 따른다 → 제목 기준 재배정

같은 샘플에서 현대해상 태아보험 글이 삼성화재·KB손해보험으로 배정된 것이 7/40 나왔다. "삼성화재 후기" 쿼리가 비교 글을 반환하기 때문이다. 정규식으로 재면 삼성화재 배정 문서의 12%(bare "KB"·"DB"를 놓친 첫 정규식 기준)이고 고친 뒤 5%였다. **제목에 브랜드가 정확히 하나 언급되고 배정 브랜드와 다르면 그 브랜드로**(40건 검수 오탐 0) 재배정하고 원래 값은 `brand_query`에 보존, 같은 규칙을 주간 파이프라인에 넣었다.

### DP13. 한 브랜드의 쏠림이 나머지 전부를 "유의"하게 만든다 → LOO·MDE·소표본 규칙

8카테고리 재검정에서 KB손해보험 상품설계 1.39×가 유의하게 나왔다. 현대해상(고객 글의 76%가 태아보험)이 others 풀을 끌어내려 나머지 4사가 전부 태아 과소·상품설계 과대로 찍힌 **one-vs-rest 거울상**이었다(현대해상 제외 시 1.05×). 검정에 세 가지를 넣었다: (1) **leave-one-out** — 경쟁사를 하나씩 빼고 재검정해 전부 유의해야 robust, `통계 검증 ✓`는 robust에만; (2) **셀별 MDE** — 비유의 셀은 "차이 없음"이 아니라 "검출 범위 밖의 차이는 없음"으로, 유의하지만 MDE 안쪽인 셀은 과대추정 경고; (3) **소표본 규칙** — n<30은 Fisher exact p와 수축 추정(m=20)·95% 구간, n<5는 미검정, 카테고리 n<30은 검정 제외. k-fold 같은 재표본은 검정력을 만들지 못하므로 표본 목표(2× 검출에 KB손해보험 자동차 220건)를 power watch로 매주 찍고, 규칙 전체는 [docs/brand_category_test_rules.md](docs/brand_category_test_rules.md)에 둔다 — 주간 누적으로 셀의 n이 임계값을 넘을 때 누가 언제 다시 보는지가 코드 밖에 있어야 하기 때문이다.

### DP14. 소급 수정되는 데이터와 DB 갱신 가드

Naver DataLab은 직전 주를 며칠 뒤 상향 수정한다(재스케일이 아니라 늦은 데이터, git diff로 확인) → 리포트 대상은 종료 후 3일 이상 지난 확정 주, 원본은 append-only 히스토리로 실측. 주간 파이프라인의 작성자 필터는 첫 dry-run에서 조건 하나(`label_source <> 'author_filter'`)가 이미 noise인 문서 2,659건을 재스탬프해 BERTopic 토픽 ID를 지웠고, 커밋된 export CSV(git HEAD)로 복구했다. 이후 규칙: DB를 바꾸는 단계는 **상태가 실제로 바뀌는 행만** 대상으로 하고(`topic_label <> 'noise'`), 영향 건수를 먼저 세어 예상 범위(신규 문서 수 + 여유)를 넘으면 중단하며, dry-run은 파생값(감성)을 쓰지 않는다.

---

## Results (2026-11-04)

### 카테고리 분포 (staging 14,899건)

| 카테고리 | 건수 | 비율 |
|---|---|---|
| 브랜드 캠페인 | 3,938 | 26.4% |
| 이벤트·프로모션 | 1,539 | 10.3% |
| 상품 설계·고객후기 | 1,059 | 7.1% |
| 자동차보험 | 1,050 | 7.0% |
| 실손·건강보험 | 944 | 6.3% |
| 태아·어린이보험 | 746 | 5.0% |
| 보험금 청구·보상 | 243 | 1.6% |
| noise | 5,380 | 36.1% |

noise에는 토픽 noise 외에 작성자 무관(unrelated) 규칙으로 편입된 blog·cafe 1,892건이 포함된다.

### 작성자 구성 (blog·cafe 8,683건)

| customer | advisor | media | unrelated |
|---|---|---|---|
| 1,458 (16.8%) | 2,628 (30.3%) | 210 (2.4%) | 4,387 (50.5%) |

### 감성 (9,461건, noise 제외)

| | Positive | Neutral | Negative |
|---|---|---|---|
| 전체 | 50.2% | 43.7% | 6.0% |
| 처리 방식 | 사전 4,150 · Haiku 2,675 · 중립 확정 2,636 |

감성은 아직 작성자 미구분 집계이며, customer-only 재집계는 다음 단계다.

### 브랜드×카테고리 robust 유의 셀 (Bonferroni + leave-one-out)

| 세그먼트 | 브랜드 | 과대 | 과소 |
|---|---|---|---|
| web_customer (n=900) | 현대해상 | 태아·어린이보험 5.57× | 상품 설계 0.30× · 실손·건강 0.24× |
| | **KB손해보험** | — | — (상품설계 1.38× 범위 내 동일, 실손·자동차 미확정) |
| web_advisor (n=1,001) | **KB손해보험** | 자동차보험 1.88× | 태아·어린이보험 0.32× |
| | 현대해상 | 태아·어린이보험 4.82× | 상품 설계 0.19× · 실손 0.18× · 자동차 0.15× |
| | 메리츠화재 | 상품 설계·고객후기 2.72× | 자동차보험 0.21× |
| youtube (n=5,608) | 삼성화재 | 이벤트·프로모션 2.51× | 브랜드 캠페인 0.50× |
| | 현대해상 | 브랜드 캠페인 1.30× | 이벤트 0.54× · 상품 설계 0.06× |
| | DB손해보험 | 상품 설계·고객후기 6.30× | 브랜드 캠페인 0.91× |

유의하지만 LOO에 취약한 셀(DB손해보험·삼성화재 자동차 과대, 4사 태아 과소 등)은 현대해상 쏠림의 거울상으로 발견에 넣지 않는다.

### 서빙 모델

| 모델 | 구성 | 지표 |
|---|---|---|
| 카테고리 분류기 | knn10, ko-sroberta 임베딩, 8 클래스, 학습 10,718건 | CV 95.6% · conf ≥ 0.7: coverage 89%, 정확도 99.2% |
| 작성자 분류기 | 소스별 knn20, 3 클래스(media→advisor), 학습 8,683건 | 게이트 적용 coverage 71% · customer precision 92.0% · unrelated precision 98.1% |

### 이상탐지 (26 events, 2/3 methods)

5사 동시 급락 주: 2024-09-16 · 2025-01-27 · 2025-10-06 · 2026-09-21 (추석·설). 복수 브랜드 동시 급등: 2025-12-08 · 2026-02-23.

### Lead-lag

VoC 볼륨 기반이라 설계사 콘텐츠·광고 물량 교란이 있어 `정성 시그널`로 해석 보류. customer-only 볼륨으로 재실행 예정.

---

## Project Structure

```
cx_reporting/
|-- config/
|   |-- settings.py              # 5사 브랜드·키워드·채널 ID·TARGET_BRAND (단일 진실원)
|   |-- db.py                    # Supabase 우선, 로컬 Docker fallback
|-- collectors/                  # naver_datalab (upsert + history), youtube, naver_voc
|-- etl/
|   |-- staging/text_preprocessing.py
|   |-- mart/{brand_search_weekly, content_performance, voc_sentiment}.py
|-- analysis/
|   |-- topic_modeling.py        # Stage 1 BERTopic × LDA (학습 전용)
|   |-- topic_modeling_stage2.py # 계층 재클러스터링
|   |-- reduce_outliers.py       # Stage 3 centroid 유사도 재배정
|   |-- topic_mapping.py         # 수동 매핑 (규칙·분류기 라벨 보호)
|   |-- embedding_cache.py       # id 키 임베딩 캐시 (학습·서빙 공용)
|   |-- author_features.py       # 작성자 분류기 피처·소스별 kNN
|   |-- author_llm.py            # 작성자 Haiku 2단계 프롬프트·Sonnet 검증
|   |-- brand_rules.py           # 제목 기준 브랜드 귀속 규칙
|   |-- sentiment.py             # 손해보험 사전 + Claude Haiku
|   |-- signal_matrix.py         # 매트릭스·층화 검정(MDE·LOO·Fisher·수축)·lead-lag·대표 문서
|   |-- anomaly_detection.py
|-- orchestration/
|   |-- weekly_run.py            # 주간 오케스트레이터 (settle 가드, dry-run)
|   |-- incremental.py           # stage / brand rule / classify / author / filter / tie-break / sentiment
|   |-- snapshot.py              # 주간 스냅샷 + 전주 대비 변화 + 알림
|-- report/weekly_report.py      # Claude Sonnet → PDF, 세그먼트·시그널 태그
|-- dashboard/app.py, tabs/
|-- scripts/
|   |-- build_serve_classifier.py, build_author_classifier.py
|   |-- relabel_catchall.py, relabel_product_bucket.py, relabel_fetal_child.py
|   |-- reassign_brand_by_title.py, label_author_type.py
|   |-- run_pre_eda.py, check_data_quality.py, inspect_noise.py, generate_figures.py, init_schema.sql
|-- docs/brand_category_test_rules.md   # 검정 판정 규칙·승격 조건·주간 재평가 트리거
|-- data/exports/                # CSV (Streamlit Cloud fallback) + PDF + 주간 run log
|-- models/serve/                # 서빙 분류기 2종 + 임베딩 캐시 (커밋)
|-- models/bertopic_*            # (gitignored, 547MB) — topic_modeling.py로 재생성
```

---

## How to Run

```bash
cp .env.example .env            # NAVER / YOUTUBE / ANTHROPIC 키, SUPABASE_DB_URL(선택)
docker-compose up -d            # 로컬 PostgreSQL :5438
psql -h localhost -p 5438 -U cx_admin -d cx_monitor -f scripts/init_schema.sql

# 초기 적재·학습 (1회)
python scripts/run_pre_eda.py
python -m collectors.naver_datalab
python -m collectors.youtube --max-videos 100
python -m collectors.naver_voc --max-per-query 500
python -m etl.staging.text_preprocessing
python -m analysis.topic_modeling
python -m analysis.topic_modeling_stage2 --parent-topic 0   --min-topic-size 40 --offset 100
python -m analysis.topic_modeling_stage2 --parent-topic 100 --min-topic-size 30 --offset 200
python -m analysis.reduce_outliers --threshold 0.65
python -m analysis.topic_mapping
python -m scripts.relabel_fetal_child --apply
python -m scripts.reassign_brand_by_title --apply
python -m scripts.label_author_type && python -m scripts.label_author_type --verify
python -m scripts.build_serve_classifier
python -m scripts.build_author_classifier
python -m analysis.sentiment

# 분석·리포트
python -m etl.mart.brand_search_weekly && python -m etl.mart.content_performance && python -m etl.mart.voc_sentiment
python -m analysis.signal_matrix
python -m analysis.anomaly_detection
python -m report.weekly_report --dry-run       # 프롬프트만
python -m report.weekly_report

# 주간 자동화
python -m orchestration.weekly_run --skip-collect --no-llm --dry-run   # 경로 점검 (과금 없음)
python -m orchestration.weekly_run

streamlit run dashboard/app.py
```

---

## Tech Stack

Python 3.10 · PostgreSQL 16 (Supabase) · BERTopic · ko-sroberta-multitask · scikit-learn (kNN, Isolation Forest) · scipy · statsmodels (MSTL) · Claude Haiku 4.5 / Sonnet 4.6 · Streamlit · Plotly · ReportLab · Naver DataLab API · YouTube Data API v3

---

*Independent Project · GitHub: [ChangyeolAidenOh](https://github.com/ChangyeolAidenOh)*
