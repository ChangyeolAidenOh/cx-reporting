"""Signal-strength tags used in exports, the weekly report and the dashboard.

Single source of truth: change a tag here and every CSV, prompt and caption follows.
Promotion conditions for TAG_STAT are in docs/brand_category_test_rules.md.
"""

TAG_STAT = "통계 검증 패스"     # Bonferroni significant AND robust to leave-one-out
TAG_CLUSTER = "cluster 시그널"  # pattern observed, not asserted
TAG_QUAL = "정성 시그널"        # quotes and qualitative evidence
TAG_LOWN = "표본 부족"          # cell too small to decide (미확정)

ALL_TAGS = [TAG_STAT, TAG_CLUSTER, TAG_QUAL, TAG_LOWN]
