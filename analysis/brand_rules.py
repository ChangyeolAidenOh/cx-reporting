"""Title-based brand resolution for blog/cafe docs.

The collection query gives each doc a brand ("삼성화재 후기" -> samsung_fire), but
comparison-heavy areas return posts that are about another company. Rule:
exactly one brand named in the title and it is not the assigned brand -> that
brand (brand_source='title'); two or more named -> keep; none -> keep.
Shared by scripts/reassign_brand_by_title.py (backfill) and
orchestration/incremental.py (weekly).
"""

import re

BRAND_PATTERNS = {
    "kb": r"(?<![a-z])kb(?![a-z])|케이비손해",
    "samsung_fire": r"삼성 ?화재|삼성 ?다이렉트",
    "db": r"(?<![a-z])db(?![a-z])|디비손해",
    "hyundai": r"현대 ?해상",
    "meritz": r"메리츠",
}
COMPILED = {b: re.compile(p, flags=re.I) for b, p in BRAND_PATTERNS.items()}


def mentioned(title):
    title = title or ""
    return [b for b, rx in COMPILED.items() if rx.search(title)]


def resolve_brand(brand, title):
    """Return (brand_to_use, changed)."""
    m = mentioned(title)
    if len(m) == 1 and m[0] != brand:
        return m[0], True
    return brand, False
