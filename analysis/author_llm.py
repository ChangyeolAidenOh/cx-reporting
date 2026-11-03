"""LLM labeling of author_type, shared by the backfill script
(scripts/label_author_type.py) and the weekly pipeline (orchestration/incremental.py).

author_type answers "who wrote this?", orthogonal to topic_label ("what is it about?").

  customer   1인칭 고객 경험·질문·불만·후기. 캠페인 영상에 단 일반 시청자 댓글 포함
  advisor    보험설계사·GA·손해사정사·대리점의 영업·안내 글 (상담 유도, 연락처, 설계안 제시, 모집종사자)
  media      언론 기사, 보험사·협회·기관 공식 안내, 채용·강의·IR 등 업무성 글
  unrelated  보험과 무관한 광고·글 (대리운전·정비·가전·부동산·맛집·주식 등; 보험은 스쳐 언급)

The prompt is two-step (is insurance the subject? -> who wrote it?) with a
structured insurance_topic field that is enforced in code. The JSON array is
extracted as the first [...] block because Haiku sometimes appends prose.
"""

import os
import re
import json
import time

LABELS = ["customer", "advisor", "media", "unrelated"]
BATCH = 25
HAIKU_MODEL = os.getenv("LLM_MODEL_SENTIMENT", "claude-haiku-4-5-20251001")
SONNET_MODEL = os.getenv("LLM_MODEL_REPORT", "claude-sonnet-4-6")

PROMPT = """손해보험 관련 웹 텍스트의 **작성자 유형**을 분류합니다. 두 단계로 판단하세요.

[1단계] 보험(가입/보장/보험료/청구/보상/설계사/보험사 앱/서비스)이 이 글의 **주제**인가?
- 글 앞부분(제목)에 보험 관련어가 없고 본문에 한두 문장만 나오면 -> unrelated.
- 다음은 모두 unrelated: 배구·야구 직관 후기(KB손해보험스타즈·삼성화재블루팡스), 여행 준비 체크리스트에 여행자보험 한 줄,
  자격증 / 시험 후기, 보험사 건물에서 열린 행사 후기, 출산 / 육아 일기에 태아등재 언급, 적금/대출/카드/주식 글.
- 1인칭 후기 형식이어도, 보험사 이름이 나와도, 주제가 보험이 아니면 unrelated.
- 예 -> 2단계로.

[2단계] 보험이 주제인 글은 누가 썼는가?
- customer: 일반 고객의 1인칭 경험/질문/불만/후기. "가입했는데", "청구했더니", "어떻게 하나요?", "쓰지 마세요". 캠페인 영상의 시청자 댓글, 짧은 응원/감탄 댓글도 customer.
- advisor: 보험설계사/GA/손해사정사/대리점/비교사이트가 쓴 영업/안내/정보 글. 상담 유도, 연락처, 설계안·추천플랜, 상품 비교표, "모집종사자", 고객센터 번호/청구서류 정리 같은 SEO 정보글 포함.
- media: 언론 기사, 보험사/협회/금감원 공식 발표, 채용공고, 강의/교육/IR/실적 등 업무성 글.
- 손해사정사·설계사가 "의뢰인/고객 후기"를 대신 올린 글도 advisor (고객이 직접 쓴 게 아님).

주의:
- 보험사가 아닌 업체(정비/수리/여행사/이사)의 광고에 보험이 언급되면 unrelated (advisor 아님).
- 보험사 스포츠팀(배구단/야구단) 경기 글은 unrelated.
- 정보성 블로그는 대부분 advisor. media는 언론/공식기관으로 좁게.

각 텍스트마다 1단계 답을 먼저 쓰고 유형을 쓰는 것으로. JSON 배열만 출력:
[{{"id":1,"insurance_topic":true,"type":"customer"}}, {{"id":2,"insurance_topic":false,"type":"unrelated"}}, ...]
insurance_topic이 false면 type은 반드시 unrelated.

텍스트:
{block}"""


def get_client():
    import anthropic
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def parse_response(text, ids):
    """Map doc ids (list, in batch order) to labels from the model's JSON array."""
    txt = re.sub(r"```json|```", "", text).strip()
    m = re.search(r"\[.*\]", txt, re.S)
    if not m:
        raise ValueError("no JSON array in response")
    out = {}
    for item in json.loads(m.group(0)):
        k = int(item["id"]) - 1
        t = str(item.get("type", "")).strip().lower()
        if not item.get("insurance_topic", True):
            t = "unrelated"
        if 0 <= k < len(ids) and t in LABELS:
            out[int(ids[k])] = t
    return out


def label_batch(client, ids, texts, model):
    block = "\n".join(f"[{j + 1}] {t[:320]}" for j, t in enumerate(texts))
    r = client.messages.create(model=model, max_tokens=900, temperature=0,
                               messages=[{"role": "user", "content": PROMPT.format(block=block)}])
    return parse_response(r.content[0].text, ids)


def label_docs(df, model=HAIKU_MODEL, batch=BATCH, retries=3, sleep=0.3, progress_every=20):
    """df needs columns id, text_clean. Returns {id: label}; ids missing from the
    dict failed after retries (caller re-runs later)."""
    client = get_client()
    out = {}
    t0 = time.time()
    n_batches = (len(df) + batch - 1) // batch
    for bi, i in enumerate(range(0, len(df), batch)):
        b = df.iloc[i:i + batch]
        ids, texts = b["id"].tolist(), b["text_clean"].tolist()
        for attempt in range(retries):
            try:
                out.update(label_batch(client, ids, texts, model))
                break
            except Exception as e:
                if attempt == retries - 1:
                    print(f"  [FAIL] batch {bi + 1}/{n_batches}: {e}")
                time.sleep(2 * (attempt + 1))
        if progress_every and bi % progress_every == 0 and bi:
            print(f"  {len(out)} labeled, {time.time() - t0:.0f}s")
        time.sleep(sleep)
    return out


def label_author_docs(df):
    """Haiku pass (weekly review and backfill)."""
    return label_docs(df, model=HAIKU_MODEL)


def verify_customer_docs(df):
    """Sonnet re-check of Haiku 'customer' labels; returns the full relabel dict."""
    return label_docs(df, model=SONNET_MODEL, progress_every=0)
