"""Resolve BODY_25 / hand / face checkpoint files.

By default, missing weights are downloaded from
``hohs/openpose135-weights`` into ``checkpoints/openpose135``. Pass
``cache_dir=`` / ``--weights_dir`` to use another local checkpoint folder, or
pass ``repo_id=`` / ``--hf_repo`` for a different Hugging Face mirror that hosts
all required files.

The HF repo must contain three files at its root:
    body_pose_model_25.pth    # caffemodel2pytorch port of pose_iter_584000.caffemodel
    hand_pose_model.pth       # CMU hand model (matches lllyasviel/Annotators)
    facenet.pth               # CMU face FaceNet (matches lllyasviel/Annotators)

NOTE: the CMU OpenPose weights are licensed for non-commercial use only.
Hosting them on a public mirror inherits that restriction.
"""
from __future__ import annotations

import shutil
from pathlib import Path

DEFAULT_HF_REPO = "hohs/openpose135-weights"
DEFAULT_CHECKPOINT_DIR = Path("checkpoints/openpose135")

FILES = {
    "body25": "body_pose_model_25.pth",
    "hand": "hand_pose_model.pth",
    "face": "facenet.pth",
}


def resolve_weights(
    repo_id: str | None = None,
    cache_dir: str | None = None,
    kinds: tuple[str, ...] | list[str] | set[str] | None = None,
) -> dict[str, str]:
    """Return a {kind: local_path} dict, downloading missing weights if needed.

    Skips downloads when a local file at `<cache_dir>/<filename>` already exists.
    Without `repo_id`, the project mirror ``hohs/openpose135-weights`` is used.
    """
    repo_id = repo_id or DEFAULT_HF_REPO
    cache_path = Path(cache_dir) if cache_dir is not None else DEFAULT_CHECKPOINT_DIR
    cache_path.mkdir(parents=True, exist_ok=True)

    out: dict[str, str] = {}
    requested = tuple(FILES.keys() if kinds is None else kinds)
    missing = [kind for kind in requested if not (cache_path / FILES[kind]).exists()]

    for kind in requested:
        fname = FILES[kind]
        local = cache_path / fname
        if local.exists():
            out[kind] = str(local)
            continue
        # Lazy import — keeps huggingface_hub off the hot path for users who pass
        # weight_paths= explicitly.
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=repo_id, filename=fname, cache_dir=str(cache_path))
        if Path(path) != local:
            shutil.copy2(path, local)
        out[kind] = str(local)
    return out
