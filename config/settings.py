"""Project-wide settings and constants."""

# Target brands for competitive analysis: 손해보험 5사
# Target brand = kb (KB손해보험), competitors = 삼성화재·DB손해보험·현대해상·메리츠화재
# Pre-EDA results
# Naver DataLab SoV: 삼성화재 22.9% / KB손해보험 22.0% / 현대해상 21.7% / DB손해보험 20.5% / 메리츠화재 12.8%
# YouTube: all 5 brands have official channels (메리츠화재 small, 6.7K subs)
# Blog VoC: all brands >= 200 monthly (sufficient)

TARGET_BRAND = "kb"

BRANDS = {
    "kb": {
        "name_kr": "KB손해보험",
        "name_en": "KB Insurance",
        "naver_keywords": ["KB손해보험", "KB손보", "케이비손해보험"],
        "youtube_channel_id": "UCWDLm80rnSGH84Wg1BCIHBQ",
        "youtube_available": True,
        "primary_voc_source": "blog+youtube",
    },
    "samsung_fire": {
        "name_kr": "삼성화재",
        "name_en": "Samsung Fire",
        "naver_keywords": ["삼성화재", "삼성화재보험"],
        "youtube_channel_id": "UCffY-gSI6eOHHS2Z146o4qQ",
        "youtube_available": True,
        "primary_voc_source": "blog+youtube",
    },
    "db": {
        "name_kr": "DB손해보험",
        "name_en": "DB Insurance",
        "naver_keywords": ["DB손해보험", "DB손보", "디비손해보험"],
        "youtube_channel_id": "UCxhDNs6XZB-Afu4BSdBGKJw",
        "youtube_available": True,
        "primary_voc_source": "blog+youtube",
    },
    "hyundai": {
        "name_kr": "현대해상",
        "name_en": "Hyundai Marine",
        "naver_keywords": ["현대해상", "현대해상화재"],
        "youtube_channel_id": "UCLdiWtQQZvuj4u15COsjnJg",
        "youtube_available": True,
        "primary_voc_source": "blog+youtube",
    },
    "meritz": {
        "name_kr": "메리츠화재",
        "name_en": "Meritz Fire",
        "naver_keywords": ["메리츠화재", "메리츠보험"],
        "youtube_channel_id": "UCqnmhJo2CEquIdxM7WIutFQ",
        "youtube_available": True,
        "primary_voc_source": "blog+youtube",
    },
}

# Korean name -> brand key (used in staging unification)
BRAND_KR_TO_KEY = {info["name_kr"]: key for key, info in BRANDS.items()}

# Naver DataLab config
# NOTE: DataLab allows max 5 keyword groups per request -> exactly 5 brands
NAVER_DATALAB = {
    "start_date": "2024-01-01",
    "end_date": "",  # set dynamically
    "time_unit": "week",
    "ages": [],
    "gender": "",
}

# YouTube API config
YOUTUBE = {
    "max_results_per_channel": 50,
    "max_comments_per_video": 100,
    "order": "date",
}

# VoC search keywords (Naver Blog/Cafe) — 손해보험 도메인 (구어체 포함, 수집 커버리지 목적)
VOC_KEYWORDS = [
    "KB손해보험 후기",
    "삼성화재 후기",
    "DB손해보험 후기",
    "현대해상 후기",
    "메리츠화재 후기",
    "KB손해보험 자동차보험",
    "KB손해보험 실손보험",
    "KB손해보험 보험금 청구",
    "KB손해보험 다이렉트",
    "자동차보험 비교",
    "실손보험 추천",
]

# Engagement score weights
ENGAGEMENT_WEIGHTS = {
    "likes": 1.0,
    "comments": 2.0,
    "shares": 3.0,
}

# Anomaly detection thresholds
ANOMALY = {
    "zscore_threshold": 2.5,
    "isolation_forest_contamination": 0.05,
    "min_methods_agree": 2,
}

# PostgreSQL schema names
SCHEMA = {
    "raw": "raw",
    "staging": "staging",
    "mart": "mart",
}
