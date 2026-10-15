#!/usr/bin/env python3
"""Download ViMoLa checkpoints from Hugging Face Hub."""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

DEFAULT_REPO = "bishoygaloaa/ViMoLa"
FILES = [
    "checkpoints/tokenizer.pt",
    "checkpoints/t2m.pt",
    "checkpoints/adapters.pt",
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--out", default=".", help="Repo-style root (creates ./checkpoints/)")
    p.add_argument("--token", default=None)
    args = p.parse_args()
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)

    for rel in FILES:
        path = hf_hub_download(
            repo_id=args.repo,
            filename=rel,
            local_dir=str(root),
            token=args.token,
        )
        print(f"✓ {path}")
    print(f"\nWeights ready under {root / 'checkpoints'}")


if __name__ == "__main__":
    main()
