-- Insurance CX Reporting: PostgreSQL Schema Init
-- 3-tier: raw -> staging -> mart

-- Create schemas
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS mart;

-- ============================================================
-- RAW LAYER: as-collected from APIs
-- ============================================================

-- Naver DataLab search trends (weekly)
CREATE TABLE IF NOT EXISTS raw.naver_datalab (
    id              SERIAL PRIMARY KEY,
    brand           VARCHAR(50) NOT NULL,
    period          DATE NOT NULL,
    ratio           NUMERIC(6, 2),
    collected_at    TIMESTAMP DEFAULT NOW(),
    UNIQUE (brand, period)
);

-- YouTube videos
CREATE TABLE IF NOT EXISTS raw.youtube_videos (
    id              SERIAL PRIMARY KEY,
    video_id        VARCHAR(20) UNIQUE NOT NULL,
    channel_id      VARCHAR(30) NOT NULL,
    brand           VARCHAR(50) NOT NULL,
    title           TEXT,
    description     TEXT,
    published_at    TIMESTAMP,
    view_count      INTEGER DEFAULT 0,
    like_count      INTEGER DEFAULT 0,
    comment_count   INTEGER DEFAULT 0,
    collected_at    TIMESTAMP DEFAULT NOW()
);

-- YouTube comments
CREATE TABLE IF NOT EXISTS raw.youtube_comments (
    id              SERIAL PRIMARY KEY,
    comment_id      VARCHAR(30) UNIQUE NOT NULL,
    video_id        VARCHAR(20) NOT NULL REFERENCES raw.youtube_videos(video_id),
    author          VARCHAR(200),
    text_original   TEXT,
    like_count      INTEGER DEFAULT 0,
    published_at    TIMESTAMP,
    collected_at    TIMESTAMP DEFAULT NOW()
);

-- Naver Blog/Cafe VoC
CREATE TABLE IF NOT EXISTS raw.naver_voc (
    id              SERIAL PRIMARY KEY,
    source          VARCHAR(20) NOT NULL,  -- 'blog' or 'cafearticle'
    title           TEXT,
    description     TEXT,
    link            TEXT,
    blogger_name    VARCHAR(200),
    postdate        DATE,
    brand           VARCHAR(50),
    query           VARCHAR(200),
    collected_at    TIMESTAMP DEFAULT NOW()
);

-- Instagram posts (manual/supplementary)
CREATE TABLE IF NOT EXISTS raw.instagram_posts (
    id              SERIAL PRIMARY KEY,
    post_id         VARCHAR(50),
    brand           VARCHAR(50) NOT NULL,
    caption         TEXT,
    like_count      INTEGER DEFAULT 0,
    comment_count   INTEGER DEFAULT 0,
    post_type       VARCHAR(20),  -- image, video, carousel
    posted_at       TIMESTAMP,
    collected_at    TIMESTAMP DEFAULT NOW()
);

-- ============================================================
-- STAGING LAYER: cleaned and enriched
-- ============================================================

-- Enriched content with topic labels
CREATE TABLE IF NOT EXISTS staging.content_enriched (
    id              SERIAL PRIMARY KEY,
    source          VARCHAR(20) NOT NULL,  -- youtube, blog, cafe, instagram
    source_id       VARCHAR(50),
    brand           VARCHAR(50) NOT NULL,
    title           TEXT,
    text_clean      TEXT,
    topic_id        INTEGER,
    topic_label     VARCHAR(200),
    sentiment_score NUMERIC(5, 3),  -- -1.0 to 1.0
    sentiment_label VARCHAR(20),    -- positive, neutral, negative
    sentiment_method VARCHAR(30),   -- dictionary, llm, hybrid
    published_at    TIMESTAMP,
    processed_at    TIMESTAMP DEFAULT NOW()
);

-- ============================================================
-- MART LAYER: analysis-ready aggregates
-- ============================================================

-- Brand search trends (weekly, with WoW)
CREATE TABLE IF NOT EXISTS mart.brand_search_weekly (
    id              SERIAL PRIMARY KEY,
    brand           VARCHAR(50) NOT NULL,
    week_start      DATE NOT NULL,
    search_ratio    NUMERIC(6, 2),
    wow_change_pct  NUMERIC(8, 4),  -- week-over-week change %
    sov_pct         NUMERIC(6, 2),  -- share of voice %
    UNIQUE (brand, week_start)
);

-- Content performance by topic
CREATE TABLE IF NOT EXISTS mart.content_performance (
    id              SERIAL PRIMARY KEY,
    brand           VARCHAR(50) NOT NULL,
    week_start      DATE NOT NULL,
    source          VARCHAR(20),
    topic_id        INTEGER,
    topic_label     VARCHAR(200),
    content_count   INTEGER DEFAULT 0,
    avg_engagement  NUMERIC(8, 4),
    total_views     INTEGER DEFAULT 0,
    total_likes     INTEGER DEFAULT 0,
    total_comments  INTEGER DEFAULT 0,
    UNIQUE (brand, week_start, source, topic_id)
);

-- VoC sentiment weekly
CREATE TABLE IF NOT EXISTS mart.voc_sentiment (
    id              SERIAL PRIMARY KEY,
    brand           VARCHAR(50) NOT NULL,
    week_start      DATE NOT NULL,
    positive_ratio  NUMERIC(5, 3),
    neutral_ratio   NUMERIC(5, 3),
    negative_ratio  NUMERIC(5, 3),
    total_docs      INTEGER DEFAULT 0,
    top_keywords    JSONB,
    UNIQUE (brand, week_start)
);

-- Content-Signal matrix (topic x engagement x search interest)
CREATE TABLE IF NOT EXISTS mart.content_signal_matrix (
    id              SERIAL PRIMARY KEY,
    brand           VARCHAR(50) NOT NULL,
    topic_id        INTEGER,
    topic_label     VARCHAR(200),
    avg_engagement  NUMERIC(8, 4),
    search_signal   NUMERIC(8, 4),  -- search interest change after content
    quadrant        VARCHAR(30),    -- high-high, high-low, low-high, low-low
    sample_count    INTEGER DEFAULT 0,
    period_start    DATE,
    period_end      DATE
);

-- Anomaly log
CREATE TABLE IF NOT EXISTS mart.anomaly_log (
    id              SERIAL PRIMARY KEY,
    brand           VARCHAR(50) NOT NULL,
    detected_date   DATE NOT NULL,
    metric          VARCHAR(100),
    value           NUMERIC(12, 4),
    zscore          NUMERIC(8, 4),
    mstl_residual   NUMERIC(12, 4),
    if_score        NUMERIC(8, 4),
    methods_agreed  INTEGER,        -- out of 3
    probable_cause  TEXT,
    reviewed        BOOLEAN DEFAULT FALSE
);

-- Weekly report metadata
CREATE TABLE IF NOT EXISTS mart.weekly_report (
    id              SERIAL PRIMARY KEY,
    brand           VARCHAR(50) NOT NULL,
    week_label      VARCHAR(20) NOT NULL,  -- e.g., "2026-W20"
    report_json     JSONB,                 -- structured report data
    report_text     TEXT,                  -- LLM-generated narrative
    pdf_path        VARCHAR(500),
    generated_at    TIMESTAMP DEFAULT NOW(),
    UNIQUE (brand, week_label)
);

-- ============================================================
-- INDEXES
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_datalab_brand_period ON raw.naver_datalab(brand, period);
CREATE INDEX IF NOT EXISTS idx_yt_video_brand ON raw.youtube_videos(brand);
CREATE INDEX IF NOT EXISTS idx_yt_comment_video ON raw.youtube_comments(video_id);
CREATE INDEX IF NOT EXISTS idx_voc_brand_date ON raw.naver_voc(brand, postdate);
CREATE INDEX IF NOT EXISTS idx_content_brand_topic ON staging.content_enriched(brand, topic_id);
CREATE INDEX IF NOT EXISTS idx_search_weekly ON mart.brand_search_weekly(brand, week_start);
CREATE INDEX IF NOT EXISTS idx_perf_weekly ON mart.content_performance(brand, week_start);
CREATE INDEX IF NOT EXISTS idx_sentiment_weekly ON mart.voc_sentiment(brand, week_start);
CREATE INDEX IF NOT EXISTS idx_anomaly_brand ON mart.anomaly_log(brand, detected_date);
