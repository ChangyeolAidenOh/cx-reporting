"""Re-save fitted BERTopic models as safetensors so they can be committed.

Pickle serialization embeds the sentence-transformer (~550MB). Safetensors
stores only the cluster structure + c-TF-IDF (~few MB) and records the
embedding model by name, so weekly `transform()` reproduces the same topic ids.

Also writes models/manifest.json so the weekly pipeline knows which files
to load and which stage-2 offsets apply.

Usage:
    python scripts/export_models.py
"""

import sys
import os
import json
import shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EMB = "jhgan/ko-sroberta-multitask"
SRC = {
    "stage1":     ("models/bertopic_insurance",  None, None),
    "stage2_0":   ("models/bertopic_stage2_0",   0,    100),
    "stage2_100": ("models/bertopic_stage2_100", 100,  200),
}
DST_DIR = "models/serve"


def run():
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer

    emb = SentenceTransformer(EMB)
    os.makedirs(DST_DIR, exist_ok=True)
    manifest = {"embedding_model": EMB, "models": {}}

    for name, (src, parent, offset) in SRC.items():
        if not os.path.exists(src):
            print(f"[SKIP] {src} not found")
            continue
        dst = f"{DST_DIR}/{name}"
        if os.path.exists(dst):
            shutil.rmtree(dst)
        model = BERTopic.load(src, embedding_model=emb)
        model.save(dst, serialization="safetensors", save_ctfidf=True, save_embedding_model=EMB)
        size = sum(f.stat().st_size for f in Path(dst).rglob("*") if f.is_file()) / 1e6
        n_topics = len([t for t in model.get_topics() if t != -1])
        manifest["models"][name] = {"path": dst, "parent_topic": parent, "offset": offset,
                                    "n_topics": n_topics, "size_mb": round(size, 2)}
        print(f"Saved {name}: {n_topics} topics, {size:.1f} MB -> {dst}")

    # outlier-reassignment threshold used at training time (keep fixed in serving)
    manifest["outlier_reassign_threshold"] = 0.65
    with open(f"{DST_DIR}/manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nManifest written: {DST_DIR}/manifest.json")
    print("Add to .gitignore exception:  !models/serve/")


if __name__ == "__main__":
    run()
