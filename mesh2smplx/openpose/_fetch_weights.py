"""Fetch OpenPose-135 .pth weight files from the project Hugging Face mirror.

Usage:
    python -m mesh2smplx.openpose._fetch_weights --out checkpoints/openpose135
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from .weights import DEFAULT_HF_REPO, FILES


def fetch_openpose135_weights(
    out_dir: Path,
    kinds: tuple[str, ...] | list[str] | set[str] | None = None,
    repo_id: str = DEFAULT_HF_REPO,
) -> None:
    requested = tuple(FILES.keys() if kinds is None else kinds)
    out_dir.mkdir(parents=True, exist_ok=True)

    for kind in requested:
        fname = FILES[kind]
        target = out_dir / fname
        if target.exists():
            print(f"  ok {fname} already at {target}")
            continue
        path = hf_hub_download(repo_id=repo_id, filename=fname, cache_dir=str(out_dir))
        if Path(path) != target:
            shutil.copy2(path, target)
        print(f"  ok {fname} -> {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/openpose135"))
    parser.add_argument("--repo", default=DEFAULT_HF_REPO)
    args = parser.parse_args()

    fetch_openpose135_weights(args.out, repo_id=args.repo)

    print("\nDone. Files in", args.out)
    for path in sorted(args.out.iterdir()):
        if path.suffix in {".pth", ".pt"}:
            print(f"  {path.name}  ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
