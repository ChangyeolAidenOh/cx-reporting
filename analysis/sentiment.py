"""NLP: Hybrid sentiment analysis for insurance VoC.

Layer 1: Insurance domain sentiment dictionary (fast, transparent, cost=0)
Layer 2: Claude Haiku API for ambiguous cases (accurate, cost-controlled)
Routing: dictionary confidence < threshold -> Haiku

Usage:
    python -m analysis.sentiment
    python -m analysis.sentiment --dict-only
    python -m analysis.sentiment --threshold 0.3
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

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LLM_MODEL_SENTIMENT = os.getenv("LLM_MODEL_SENTIMENT", "claude-haiku-4-5-20251001")


# ============================================================
# Insurance domain sentiment dictionary
# ============================================================

POSITIVE_WORDS = {
    # General positive
    "만족": 1.0, "추천": 0.8, "좋다": 0.7, "좋아요": 0.7, "좋은": 0.7,
    "감사": 0.8, "감동": 0.9, "최고": 0.9, "훌륭": 0.9, "대박": 0.8,
    "응원": 0.7, "기대": 0.6, "멋지": 0.8, "뿌듯": 0.8, "행복": 0.8,
    "유익": 0.8, "도움": 0.7, "편리": 0.7, "신뢰": 0.8, "안심": 0.8,
    "재밌": 0.6, "재미있": 0.6, "웃기": 0.5, "신기": 0.5, "놀랍": 0.6,
    # Insurance-specific positive
    "든든": 0.9, "보장": 0.6, "혜택": 0.7, "절약": 0.6, "환급": 0.6,
    "수령": 0.5, "보상": 0.5, "가성비": 0.7, "저렴": 0.6, "합리적": 0.7,
    "빠른": 0.5, "친절": 0.8, "꼼꼼": 0.7, "세심": 0.7, "전문적": 0.7,
    "건강": 0.4, "예방": 0.4, "관리": 0.3,
}

NEGATIVE_WORDS = {
    # General negative
    "불만": -0.9, "실망": -0.8, "후회": -0.9, "최악": -1.0, "별로": -0.6,
    "짜증": -0.7, "화나": -0.8, "답답": -0.7, "불편": -0.7, "어렵": -0.5,
    "걱정": -0.4, "불안": -0.5, "무섭": -0.5, "힘들": -0.5, "슬프": -0.6,
    "유치": -0.5, "어설프": -0.5,
    # Insurance-specific negative
    "해지": -0.7, "사기": -1.0, "피해": -0.9, "손해": -0.8, "거절": -0.7,
    "거부": -0.7, "불친절": -0.9, "불투명": -0.8, "비싸": -0.6, "과다": -0.6,
    "거짓": -0.9, "허위": -0.9, "약관": -0.3, "분쟁": -0.8, "민원": -0.7,
    "보험료인상": -0.7, "갱신": -0.4, "부담": -0.5, "강제": -0.8,
    "설계사": -0.2, "전화": -0.2,  # context-dependent, low weight
}

NEGATION_WORDS = {"안", "못", "없", "아닌", "아니", "않", "절대"}


def dict_sentiment(text):
    """Score text using sentiment dictionary.

    Returns:
        score: float (-1.0 to 1.0)
        confidence: float (0.0 to 1.0) -- how certain the dictionary is
        matched_words: list of matched sentiment words
    """
    if not text or not isinstance(text, str):
        return 0.0, 0.0, []

    words = text.lower()
    pos_matches = []
    neg_matches = []

    # Check positive words
    for word, weight in POSITIVE_WORDS.items():
        if word in words:
            # Check for negation within 3 chars before
            idx = words.find(word)
            prefix = words[max(0, idx - 3):idx]
            if any(neg in prefix for neg in NEGATION_WORDS):
                neg_matches.append((f"NOT-{word}", -weight * 0.7))
            else:
                pos_matches.append((word, weight))

    # Check negative words
    for word, weight in NEGATIVE_WORDS.items():
        if word in words:
            idx = words.find(word)
            prefix = words[max(0, idx - 3):idx]
            if any(neg in prefix for neg in NEGATION_WORDS):
                pos_matches.append((f"NOT-{word}", -weight * 0.7))
            else:
                neg_matches.append((word, weight))

    all_matches = pos_matches + neg_matches
    if not all_matches:
        return 0.0, 0.0, []

    # Weighted average
    total_weight = sum(abs(w) for _, w in all_matches)
    score = sum(w for _, w in all_matches) / len(all_matches)
    score = max(-1.0, min(1.0, score))

    # Confidence: more matches + stronger signals = higher confidence
    match_count_factor = min(len(all_matches) / 3, 1.0)
    strength_factor = total_weight / len(all_matches)
    confidence = min(match_count_factor * strength_factor, 1.0)

    matched_words = [w for w, _ in all_matches]
    return round(score, 3), round(confidence, 3), matched_words


def score_to_label(score, confidence):
    """Convert score to sentiment label."""
    if confidence < 0.1:
        return "neutral"
    if score > 0.15:
        return "positive"
    if score < -0.15:
        return "negative"
    return "neutral"


# ============================================================
# Claude Haiku API
# ============================================================

def call_haiku_batch(texts, batch_size=20):
    """Call Claude Haiku for sentiment classification.

    Returns list of (score, label) tuples.
    """
    try:
        import anthropic
    except ImportError:
        print("[ERROR] anthropic package not installed. pip install anthropic")
        return [(0.0, "neutral")] * len(texts)

    if not ANTHROPIC_API_KEY:
        print("[ERROR] ANTHROPIC_API_KEY not set")
        return [(0.0, "neutral")] * len(texts)

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Build prompt with numbered texts
        text_block = "\n".join(
            f"[{j + 1}] {t[:300]}" for j, t in enumerate(batch)
        )

        prompt = f"""다음 보험 관련 한국어 텍스트들의 감성을 분석해주세요.
각 텍스트에 대해 JSON 배열로 응답해주세요. 다른 텍스트 없이 JSON만 출력하세요.

형식: [{{"id": 1, "label": "positive|neutral|negative", "score": -1.0~1.0}}]

텍스트:
{text_block}"""

        try:
            response = client.messages.create(
                model=LLM_MODEL_SENTIMENT,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse response
            response_text = response.content[0].text.strip()
            # Clean markdown fences if present
            response_text = re.sub(r"```json\s*", "", response_text)
            response_text = re.sub(r"```\s*", "", response_text)

            parsed = json.loads(response_text)

            for item in parsed:
                score = float(item.get("score", 0.0))
                label = item.get("label", "neutral")
                results.append((score, label))

        except Exception as e:
            print(f"  [WARN] Haiku batch error: {e}")
            results.extend([(0.0, "neutral")] * len(batch))

        # Rate limiting
        if i + batch_size < len(texts):
            time.sleep(0.5)

    return results


# ============================================================
# Hybrid pipeline
# ============================================================

def run(threshold=0.3, dict_only=False):
    """Run hybrid sentiment analysis.

    Args:
        threshold: dictionary confidence below this triggers Haiku
        dict_only: if True, skip Haiku calls entirely
    """
    # Extract: get documents eligible for sentiment analysis
    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT id, text_clean, topic_id, topic_label
            FROM staging.content_enriched
            WHERE topic_label != 'noise'
               OR topic_id = 7
            ORDER BY id
            """,
            conn,
        )

    print(f"Sentiment analysis target: {len(df)} documents")

    # Layer 1: Dictionary scoring
    print("\nLayer 1: Dictionary sentiment...")
    dict_results = []
    for _, row in df.iterrows():
        score, confidence, matched = dict_sentiment(row["text_clean"])
        label = score_to_label(score, confidence)
        dict_results.append({
            "id": row["id"],
            "score": score,
            "confidence": confidence,
            "label": label,
            "matched_words": matched,
        })

    dict_df = pd.DataFrame(dict_results)

    # Stats
    high_conf = dict_df[dict_df["confidence"] >= threshold]
    low_conf = dict_df[(dict_df["confidence"] > 0) & (dict_df["confidence"] < threshold)]
    print(f"  High confidence (>={threshold}): {len(high_conf)} ({len(high_conf)/len(df)*100:.1f}%)")
    print(f"  Low confidence (<{threshold}): {len(low_conf)} ({len(low_conf)/len(df)*100:.1f}%)")
    print(f"  Dict distribution: {dict_df['label'].value_counts().to_dict()}")

    # Layer 2: Haiku for low-confidence cases
    if not dict_only and len(low_conf) > 0:
        print(f"\nLayer 2: Claude Haiku for {len(low_conf)} ambiguous docs...")
        low_conf_texts = df[df["id"].isin(low_conf["id"])]["text_clean"].tolist()

        haiku_results = call_haiku_batch(low_conf_texts, batch_size=20)

        # Merge Haiku results
        for idx, (haiku_score, haiku_label) in zip(low_conf.index, haiku_results):
            dict_df.loc[idx, "score"] = haiku_score
            dict_df.loc[idx, "label"] = haiku_label
            dict_df.loc[idx, "method"] = "llm"

    # Set method
    dict_df["method"] = dict_df.apply(
        lambda row: "llm" if row.get("method") == "llm"
        else ("dictionary" if row["confidence"] >= threshold else "low_conf_dict"),
        axis=1,
    )

    # Final distribution
    print(f"\nFinal sentiment distribution:")
    print(f"  {dict_df['label'].value_counts().to_dict()}")
    print(f"  Methods: {dict_df['method'].value_counts().to_dict()}")

    # Update staging
    update_staging(dict_df)

    # Save CSV
    save_csv(dict_df)

    return dict_df


def update_staging(result_df):
    """Update staging.content_enriched with sentiment results."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            for _, row in result_df.iterrows():
                cur.execute(
                    """
                    UPDATE staging.content_enriched
                    SET sentiment_score = %s,
                        sentiment_label = %s,
                        sentiment_method = %s
                    WHERE id = %s
                    """,
                    (
                        float(row["score"]),
                        row["label"],
                        row["method"],
                        int(row["id"]),
                    ),
                )
    print(f"Updated {len(result_df)} rows in staging.content_enriched")


def save_csv(result_df, path="data/processed/sentiment_results.csv"):
    """Save sentiment results to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    result_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"CSV exported: {path}")


def main():
    parser = argparse.ArgumentParser(description="Hybrid sentiment analysis")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Dictionary confidence threshold for Haiku routing")
    parser.add_argument("--dict-only", action="store_true",
                        help="Skip Haiku, use dictionary only")
    args = parser.parse_args()

    run(threshold=args.threshold, dict_only=args.dict_only)


if __name__ == "__main__":
    main()
