"""
Download every model the sprint container needs into sprint/models/ so the
image runs with --network none. Run once on a networked dev machine.

    python scripts/fetch_models.py [--whisper large-v3] [--gemma]
"""

from __future__ import annotations

import argparse
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "models"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--whisper", default="large-v3", help="faster-whisper size (large-v3, distil-large-v3, medium.en)")
    ap.add_argument("--gemma", action="store_true", help="also fetch Gemma 4 E4B (gated: accept license on HF first)")
    ap.add_argument("--gemma-repo", default="google/gemma-4-E4B-it")
    a = ap.parse_args()
    from huggingface_hub import snapshot_download

    snapshot_download(f"Systran/faster-whisper-{a.whisper}", local_dir=MODELS / "whisper")
    from fastembed import TextEmbedding

    TextEmbedding("BAAI/bge-small-en-v1.5", cache_dir=str(MODELS / "fastembed"))
    if a.gemma:
        snapshot_download(a.gemma_repo, local_dir=MODELS / "gemma")
    print(f"models in {MODELS}")


if __name__ == "__main__":
    main()
