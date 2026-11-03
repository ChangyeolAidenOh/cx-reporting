"""Label author_type for every staging doc with Claude Haiku (resumable).

author_type answers "who wrote this?" — orthogonal to topic_label ("what is it about?").

  customer   1인칭 고객 경험·질문·불만·후기. 캠페인 영상에 단 일반 시청자 댓글 포함
  advisor    보험설계사·GA·손해사정사·대리점의 영업·안내 글 (상담 유도, 연락처, 설계안 제시, 모집종사자)
  media      언론 기사, 보험사·협회·기관 공식 안내, 채용·강의·IR 등 업무성 글
  unrelated  보험과 무관한 광고·글 (대리운전·정비·가전·부동산·맛집·주식 등; 보험은 스쳐 언급)

Usage:
    python scripts/label_author_type.py --limit 100        # smoke test
    python scripts/label_author_type.py                    # label all NULL rows (resumable)
    python scripts/label_author_type.py --sample 40        # print a random check sample
"""

import sys
import os
import re
import json
import time
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from dotenv import load_dotenv
from config.db import get_conn

load_dotenv()

LABELS = ["customer", "advisor", "media", "unrelated"]
BATCH = 25
MODEL = os.getenv("LLM_MODEL_SENTIMENT", "claude-haiku-4-5-20251001")

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


def ensure_column():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE staging.content_enriched ADD COLUMN IF NOT EXISTS author_type VARCHAR(20)")
            cur.execute("ALTER TABLE staging.content_enriched ADD COLUMN IF NOT EXISTS author_type_src VARCHAR(10)")


def fetch_unlabeled(limit=None):
    q = ("SELECT id, text_clean FROM staging.content_enriched "
         "WHERE author_type IS NULL AND source IN ('blog','cafearticle') ORDER BY id")
    if limit:
        q += f" LIMIT {int(limit)}"
    with get_conn() as conn:
        return pd.read_sql(q, conn)


def apply_source_rules():
    """Deterministic labels where the source already tells us the author."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE staging.content_enriched SET author_type='customer', author_type_src='rule' "
                        "WHERE source='youtube_comment' AND author_type IS NULL")
            n = cur.rowcount
    print(f"[rule] youtube_comment -> customer: {n}")


def label_batch(client, batch):
    block = "\n".join(f"[{j + 1}] {t[:320]}" for j, t in enumerate(batch["text_clean"]))
    r = client.messages.create(model=MODEL, max_tokens=900, temperature=0,
                               messages=[{"role": "user", "content": PROMPT.format(block=block)}])
    txt = re.sub(r"```json|```", "", r.content[0].text).strip()
    m = re.search(r"\[.*\]", txt, re.S)  # first [...] block only
    if not m:
        raise ValueError("no JSON array in response")
    out = {}
    for item in json.loads(m.group(0)):
        k = int(item["id"]) - 1
        t = str(item.get("type", "")).strip().lower()
        if not item.get("insurance_topic", True):
            t = "unrelated"  # enforce step 1 in code, not just in prompt
        if 0 <= k < len(batch) and t in LABELS:
            out[int(batch.iloc[k]["id"])] = t
    return out


def save(labels, src="haiku"):
    if not labels:
        return
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany("UPDATE staging.content_enriched SET author_type=%s, author_type_src=%s WHERE id=%s",
                            [(t, src, i) for i, t in labels.items()])


def run(limit=None):
    import anthropic
    ensure_column()
    apply_source_rules()
    df = fetch_unlabeled(limit)
    if df.empty:
        print("nothing to label"); return
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    print(f"Labeling {len(df)} docs in batches of {BATCH} ({(len(df) + BATCH - 1) // BATCH} calls)")

    done, failed, t0 = 0, 0, time.time()
    for i in range(0, len(df), BATCH):
        b = df.iloc[i:i + BATCH]
        for attempt in range(3):
            try:
                out = label_batch(client, b)
                save(out)
                done += len(out)
                failed += len(b) - len(out)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  [FAIL] batch {i // BATCH}: {e}")
                    failed += len(b)
                time.sleep(2 * (attempt + 1))
        if (i // BATCH) % 20 == 0:
            print(f"  {done} labeled, {failed} failed, {time.time() - t0:.0f}s")
        time.sleep(0.3)

    print(f"\nDone: {done} labeled, {failed} failed (re-run to retry)")
    with get_conn() as conn:
        dist = pd.read_sql("SELECT author_type, COUNT(*) n FROM staging.content_enriched "
                           "GROUP BY 1 ORDER BY 2 DESC", conn)
    print(dist.to_string(index=False))

def verify(limit=None):
    """Re-check Haiku 'customer' labels with Sonnet; Sonnet's answer overwrites."""
    import anthropic
    sonnet = os.getenv("LLM_MODEL_REPORT", "claude-sonnet-4-6")
    q = ("SELECT id, text_clean FROM staging.content_enriched "
         "WHERE author_type='customer' AND COALESCE(author_type_src,'haiku')='haiku' ORDER BY id")
    if limit:
        q += f" LIMIT {int(limit)}"
    with get_conn() as conn:
        df = pd.read_sql(q, conn)
    if df.empty:
        print("nothing to verify"); return
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    print(f"Verifying {len(df)} customer labels with {sonnet}")
    changed = 0
    for i in range(0, len(df), BATCH):
        b = df.iloc[i:i + BATCH]
        block = "\n".join(f"[{j + 1}] {t[:320]}" for j, t in enumerate(b["text_clean"]))
        try:
            r = client.messages.create(model=sonnet, max_tokens=900, temperature=0,
                                       messages=[{"role": "user", "content": PROMPT.format(block=block)}])
            txt = re.sub(r"```json|```", "", r.content[0].text).strip()
            out = {}
            for item in json.loads(txt):
                k = int(item["id"]) - 1
                t = str(item.get("type", "")).strip().lower()
                if not item.get("insurance_topic", True):
                    t = "unrelated"
                if 0 <= k < len(b) and t in LABELS:
                    out[int(b.iloc[k]["id"])] = t
                    if t != "customer":
                        changed += 1
            save(out, src="sonnet")
        except Exception as e:
            print(f"  [WARN] verify batch {i // BATCH}: {e}")
        time.sleep(0.3)
    print(f"Verified {len(df)}: {changed} relabeled away from customer ({changed / len(df):.0%})")


def sample(n=40):
    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT author_type, source, LEFT(text_clean, 100) t FROM (
              SELECT *, ROW_NUMBER() OVER (PARTITION BY author_type ORDER BY RANDOM()) rn
              FROM staging.content_enriched WHERE author_type IS NOT NULL) x
            WHERE rn <= %s ORDER BY author_type
            """, conn, params=(max(n // 4, 1),))
    for _, r in df.iterrows():
        print(f"[{r['author_type']:<9}|{r['source']:<15}] {r['t']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()
    if a.sample:
        sample(a.sample)
    elif a.verify:
        verify(a.limit)
    else:
        run(a.limit)


if __name__ == "__main__":
    main()
