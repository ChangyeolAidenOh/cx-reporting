"""Id-keyed ko-sroberta embedding cache for staging.content_enriched docs.

Why: the legacy cache (models/serve/train_embeddings.npy) was aligned by row
order of one SQL query, so any change to the training subset silently
misaligned it. This cache stores (ids, vectors) and encodes only ids that are
missing, so training scripts and the weekly pipeline can share it.

Files (models/serve/):
    doc_embeddings.npy      float16 [n, 768], L2-normalized
    doc_embedding_ids.npy   int64   [n]

Legacy bootstrap: the old cache was built from the initial load (all docs at
that time, ordered by id), and incremental docs are appended with higher ids,
so its row order is reconstructed as ORDER BY id LIMIT n_legacy. The old
vectors are adopted only after re-encoding a random sample and checking
cosine >= 0.99 against them.

Usage (module):
    from analysis.embedding_cache import get_embeddings
    X = get_embeddings(ids, texts)   # float32, aligned to ids
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from config.db import get_conn

EMB_NAME = "jhgan/ko-sroberta-multitask"
CACHE_DIR = "models/serve"
EMB_FILE = f"{CACHE_DIR}/doc_embeddings.npy"
IDS_FILE = f"{CACHE_DIR}/doc_embedding_ids.npy"
LEGACY_FILE = f"{CACHE_DIR}/train_embeddings.npy"

_model = None


def _encoder():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMB_NAME)
    return _model


def encode(texts, batch_size=64):
    return _encoder().encode(list(texts), batch_size=batch_size, show_progress_bar=True,
                             normalize_embeddings=True).astype(np.float32)


def _load():
    if os.path.exists(EMB_FILE) and os.path.exists(IDS_FILE):
        return np.load(IDS_FILE), np.load(EMB_FILE)
    return np.zeros(0, dtype=np.int64), np.zeros((0, 768), dtype=np.float16)


def _save(ids, X):
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(IDS_FILE, ids.astype(np.int64))
    np.save(EMB_FILE, X.astype(np.float16))


def bootstrap_from_legacy(sample_n=50, min_cos=0.99, seed=42):
    """Adopt the legacy positional cache if its row order can be reconstructed and verified."""
    if os.path.exists(EMB_FILE) or not os.path.exists(LEGACY_FILE):
        return False
    legacy = np.load(LEGACY_FILE)
    with get_conn() as conn:
        df = pd.read_sql("SELECT id, text_clean FROM staging.content_enriched "
                         "ORDER BY id LIMIT %s", conn, params=(int(len(legacy)),))
    if len(df) != len(legacy):
        print(f"[cache] legacy rows {len(legacy)} != query rows {len(df)}, not adopting")
        return False
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(df), size=min(sample_n, len(df)), replace=False)
    fresh = encode(df["text_clean"].iloc[pick].tolist())
    cos = (fresh * legacy[pick].astype(np.float32)).sum(1)
    print(f"[cache] legacy verification on {len(pick)} docs: cosine min={cos.min():.4f} mean={cos.mean():.4f}")
    if cos.min() < min_cos:
        print("[cache] verification failed, not adopting legacy cache")
        return False
    _save(df["id"].values, legacy)
    print(f"[cache] adopted legacy cache: {len(df)} docs")
    return True


def get_embeddings(ids, texts):
    """Return float32 embeddings aligned to ids; encodes and caches missing ids."""
    ids = np.asarray(ids, dtype=np.int64)
    texts = list(texts)
    if len(ids) != len(texts):
        raise ValueError("ids and texts length mismatch")
    bootstrap_from_legacy()
    c_ids, c_X = _load()
    pos = {int(i): k for k, i in enumerate(c_ids)}
    missing = [k for k, i in enumerate(ids) if int(i) not in pos]
    if missing:
        print(f"[cache] encoding {len(missing)} new docs ({len(ids) - len(missing)} cached)")
        new_X = encode([texts[k] for k in missing])
        c_ids = np.concatenate([c_ids, ids[missing]])
        c_X = np.concatenate([c_X, new_X.astype(np.float16)])
        _save(c_ids, c_X)
        pos = {int(i): k for k, i in enumerate(c_ids)}
    else:
        print(f"[cache] all {len(ids)} docs cached")
    idx = np.array([pos[int(i)] for i in ids])
    return c_X[idx].astype(np.float32)


if __name__ == "__main__":
    ok = bootstrap_from_legacy()
    c_ids, c_X = _load()
    print(f"cache: {len(c_ids)} docs, {c_X.shape}, bootstrap={'yes' if ok else 'no'}")
