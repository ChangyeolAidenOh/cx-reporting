"""Feature helpers for the author_type classifier, shared by
scripts/build_author_classifier.py and orchestration/incremental.py.

Embeddings come from analysis.embedding_cache; this module adds the
hand-made lexical block and the source flag, and defines the per-source
kNN wrapper so the serving code never has to know which variant was picked.
"""

import re
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

LEX_PATTERNS = {
    "phone": r"\d{2,3}-\d{3,4}-\d{4}",
    "url": r"https?://|www\.|\.com|\.kr",
    "ad": r"무료상담|상담신청|문의주세요|바로가기|연락처|카톡|카카오톡|오픈채팅|DM|링크",
    "promo": r"추천|비교|순위|꿀팁|총정리|정리해|확인하세요|알아보세요|알려드|소개해|안내",
    "first_person": r"저는|제가|저희|우리 ?아이|했어요|했는데|했습니다|했네요|궁금해요|고민",
    "question": r"\?|인가요|일까요|할까요|있을까요|계신가요",
    "media": r"기자|뉴스|보도|보도자료|공시|발표했다|밝혔다|전했다",
    "job": r"채용|모집|구인|신입|경력직|지원자",
}
SOURCE_FLAG = {"blog": 1.0, "cafearticle": 0.0}


def lexical_features(titles, texts):
    rows = []
    for ti, tx in zip(titles, texts):
        ti = ti or ""
        tx = tx or ""
        both = ti + " " + tx
        n = max(len(tx), 1)
        f = [np.log1p(len(tx)), np.log1p(len(ti))]
        for pat in LEX_PATTERNS.values():
            c = len(re.findall(pat, both))
            f.append(np.log1p(c))
            f.append(c / n * 1000)
        f.append(1.0 if re.search(LEX_PATTERNS["promo"], ti) else 0.0)
        rows.append(f)
    return np.array(rows, dtype=np.float32)


def source_flag(sources):
    return np.array([[SOURCE_FLAG.get(s, 0.5)] for s in sources], dtype=np.float32)


class PerSourceKNN:
    """One kNN per source (blog / cafe). Input X = [embedding | source_flag]."""

    def __init__(self, n_neighbors=10):
        self.n_neighbors = n_neighbors
        self.models = {}
        self.classes_ = None

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.array(sorted(set(y)))
        flag = X[:, -1]
        for s in np.unique(flag):
            m = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric="cosine", weights="distance")
            sel = flag == s
            m.fit(X[sel, :-1], y[sel])
            self.models[float(s)] = m
        return self

    def predict_proba(self, X):
        flag = X[:, -1]
        out = np.zeros((len(X), len(self.classes_)), dtype=np.float32)
        for s, m in self.models.items():
            sel = flag == s
            if not sel.any():
                continue
            p = m.predict_proba(X[sel, :-1])
            for j, c in enumerate(m.classes_):
                out[sel, np.searchsorted(self.classes_, c)] = p[:, j]
        missing = ~np.isin(flag, list(self.models))
        if missing.any():
            any_model = next(iter(self.models.values()))
            p = any_model.predict_proba(X[missing, :-1])
            for j, c in enumerate(any_model.classes_):
                out[missing, np.searchsorted(self.classes_, c)] = p[:, j]
        return out

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]
