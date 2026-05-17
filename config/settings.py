"""Project-wide settings and constants."""

# Target brands for competitive analysis
# YouTube channel IDs confirmed via pre-EDA (2026-05-14)
BRANDS = {
    "metlife": {
        "name_kr": "메트라이프",
        "name_en": "MetLife",
        "naver_keywords": ["메트라이프", "메트라이프생명", "메트라이프보험"],
        "youtube_channel_id": "",  # no official channel (subs 31, MDRT only)
        "youtube_available": False,
        "primary_voc_source": "blog",  # Blog VoC only (monthly ~214)
    },
    "samsung": {
        "name_kr": "삼성생명",
        "name_en": "Samsung Life",
        "naver_keywords": ["삼성생명", "삼성생명보험"],
        "youtube_channel_id": "UCAgkMrESCDJbgqS8zPHboTw",
        "youtube_available": True,
        "primary_voc_source": "blog+youtube",
    },
    "hanwha": {
        "name_kr": "한화생명",
        "name_en": "Hanwha Life",
        "naver_keywords": ["한화생명", "한화생명보험"],
        "youtube_channel_id": "UCpA6wY9xIh7IHUziGnd9kdg",
        "youtube_available": True,
        "primary_voc_source": "blog+youtube",
    },
    "kyobo": {
        "name_kr": "교보생명",
        "name_en": "Kyobo Life",
        "naver_keywords": ["교보생명", "교보생명보험"],
        "youtube_channel_id": "UCTXXn-qrL8GQtuHYJBuFznA",
        "youtube_available": True,
        "primary_voc_source": "blog+youtube",
    },
}

# Pre-EDA results summary (2026-05-14)
# Naver DataLab SoV: 삼성 42.4% / 한화 32.6% / 교보 19.8% / 메트라이프 5.2%
# MetLife YouTube: effectively unavailable (31 subs, 0 comments)
# Blog VoC: all 4 brands >= 200 monthly (sufficient)
# Cafe VoC: postdate field empty in API response, handle separately in collector

# Naver DataLab config
NAVER_DATALAB = {
    "start_date": "2024-01-01",
    "end_date": "",  # set dynamically
    "time_unit": "week",
    "ages": [],  # all ages
    "gender": "",  # all
}

# YouTube API config
YOUTUBE = {
    "max_results_per_channel": 50,
    "max_comments_per_video": 100,
    "order": "date",
}

# VoC search keywords (Naver Blog/Cafe)
VOC_KEYWORDS = [
    "메트라이프 보험 후기",
    "삼성생명 후기",
    "한화생명 후기",
    "교보생명 후기",
    "메트라이프 보험 가입",
    "메트라이프 보험 해지",
    "보험 추천",
    "보험 비교",
]

# Engagement score formula
# engagement_rate = (likes + comments + shares) / subscribers * 100
ENGAGEMENT_WEIGHTS = {
    "likes": 1.0,
    "comments": 2.0,  # comments indicate deeper engagement
    "shares": 3.0,    # shares indicate advocacy
}

# Anomaly detection thresholds
ANOMALY = {
    "zscore_threshold": 2.5,
    "isolation_forest_contamination": 0.05,
    "min_methods_agree": 2,  # out of 3 (Z-score, MSTL, IF)
}

# PostgreSQL schema names
SCHEMA = {
    "raw": "raw",
    "staging": "staging",
    "mart": "mart",
}
