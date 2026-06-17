"""AITviewer integration."""

from __future__ import annotations

from pathlib import Path

from mesh2smplx.core.config import PipelineConfig


def require_aitviewer():
    try:
        import aitviewer
    except ImportError as exc:
        raise RuntimeError(
            "Visualization requires AITviewer. Install `smpl-registration[viewer]` "
            "in the final package."
        ) from exc
    return aitviewer


def open_viewer(config: PipelineConfig, output_dir: Path) -> None:
    """Open pipeline outputs in AITviewer.

    The final implementation should load SMPL params, source meshes, cameras,
    and 3D keypoints. This stub keeps the public dependency boundary explicit.
    """
    require_aitviewer()
    raise NotImplementedError(
        "Draft only: AITviewer scene construction is not implemented yet. "
        f"Config input={config.input.mode}, outputs={output_dir}"
    )
