"""Record and replay optimizer snapshots without a live websocket viewer."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .aitviewer_camera_scene import (
    CameraImageOverlayConfig,
    InitialCameraConfig,
    ViewerRenderConfig,
    add_camera_image_overlays,
    parse_camera_ids,
    parse_frame_ids,
    set_initial_camera_view,
    update_viewer_config,
)

if TYPE_CHECKING:
    from mesh2smplx.fitting.smpl_fitter import SmplFitProgress


@dataclass
class FitProgressRecorder:
    """Collect fitting snapshots and write an AITviewer-friendly archive."""

    output_path: Path
    source_meshes: list[tuple[np.ndarray, np.ndarray]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    steps: list[int] = field(default_factory=list)
    total_steps: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    phases: list[str] = field(default_factory=list)
    vertices: list[np.ndarray] = field(default_factory=list)
    joints: list[np.ndarray] = field(default_factory=list)
    target_joints: list[np.ndarray] = field(default_factory=list)
    faces: np.ndarray | None = None

    def __call__(self, progress: "SmplFitProgress") -> None:
        self.steps.append(int(progress.step))
        self.total_steps.append(int(progress.total_steps))
        self.losses.append(float(progress.loss))
        self.phases.append(str(progress.phase))
        self.vertices.append(_to_float32(progress.vertices))
        self.joints.append(_to_float32(progress.joints))
        self.target_joints.append(_to_float32(progress.target_joints))
        self.faces = np.asarray(progress.faces, dtype=np.int32)

    @property
    def recorded_count(self) -> int:
        return len(self.vertices)

    def save(self) -> Path | None:
        if not self.vertices:
            return None

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {
            "vertices": np.stack(self.vertices).astype(np.float32, copy=False),
            "joints": np.stack(self.joints).astype(np.float32, copy=False),
            "target_joints": np.stack(self.target_joints).astype(np.float32, copy=False),
            "faces": np.asarray(self.faces, dtype=np.int32),
            "steps": np.asarray(self.steps, dtype=np.int32),
            "total_steps": np.asarray(self.total_steps, dtype=np.int32),
            "losses": np.asarray(self.losses, dtype=np.float32),
            "phases": np.asarray(self.phases),
            "metadata_json": np.asarray(json.dumps(self.metadata, sort_keys=True)),
        }
        self._add_source_mesh_arrays(arrays)
        np.savez_compressed(self.output_path, **arrays)
        print(f"recorded_fit_progress={self.output_path} snapshots={self.recorded_count}")
        return self.output_path

    def _add_source_mesh_arrays(self, arrays: dict[str, Any]) -> None:
        if not self.source_meshes:
            return
        vertices = [np.asarray(item[0], dtype=np.float32) for item in self.source_meshes]
        faces = [np.asarray(item[1], dtype=np.int32) for item in self.source_meshes]
        if len(vertices) == 1:
            arrays["source_vertices"] = vertices[0]
            arrays["source_faces"] = faces[0]
            return
        if (
            all(vertex.shape == vertices[0].shape for vertex in vertices)
            and all(face.shape == faces[0].shape for face in faces)
        ):
            arrays["source_vertices"] = np.stack(vertices)
            arrays["source_faces"] = np.stack(faces)


def launch_recording_viewer(
    recording_path: Path,
    *,
    batch_index: int = 0,
    source_index: int = 0,
    show_source: bool = True,
    show_joints: bool = True,
    show_cameras: bool = True,
    camera_json: Path | None = None,
    image_root: Path | None = None,
    frame_ids: tuple[int, ...] | None = None,
    camera_ids: tuple[str, ...] | None = None,
    max_cameras: int = 4,
    camera_scale: float = 0.001,
    initial_camera: str | None = "auto",
    billboard_distance: float = 0.5,
    billboard_alpha: float = 0.55,
    image_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    floor_enabled: bool = False,
    render_config: ViewerRenderConfig | None = None,
) -> None:
    """Open a saved optimizer recording in AITviewer."""
    from aitviewer.renderables.meshes import Meshes
    from aitviewer.viewer import Viewer
    from aitviewer.configuration import CONFIG as C

    try:
        from aitviewer.renderables.spheres import Spheres
    except ImportError:
        Spheres = None

    data = np.load(recording_path, allow_pickle=False)
    metadata = _load_metadata(data)
    vertices = _select_batch(data["vertices"], batch_index)
    faces = np.asarray(data["faces"], dtype=np.int32)
    camera_json = _resolve_optional_path(camera_json, metadata.get("camera_json"))
    image_root = _resolve_optional_path(image_root, metadata.get("image_root"))
    if frame_ids is None:
        frame_ids = _metadata_frame_ids(metadata)

    update_viewer_config(
        C,
        server_enabled=False,
        server_port=8417,
        render_config=render_config or ViewerRenderConfig(),
    )
    viewer = Viewer()
    _bind_legacy_render_loop(viewer)
    viewer.scene.floor.enabled = floor_enabled
    added_cameras: dict[str, Any] = {}

    if show_cameras and camera_json is not None:
        overlay_frame_ids = frame_ids or (0,)
        added_cameras = add_camera_image_overlays(
            viewer,
            CameraImageOverlayConfig(
                camera_json=camera_json,
                image_root=image_root or Path("__missing_camera_images__"),
                frame_ids=overlay_frame_ids,
                camera_ids=camera_ids,
                max_cameras=max_cameras,
                camera_scale=camera_scale,
                billboard_distance=billboard_distance,
                billboard_alpha=billboard_alpha,
                image_extensions=image_extensions,
            ),
        )
        print(f"Added {len(added_cameras)} calibrated cameras for reprojection.")

    if show_source and "source_vertices" in data and "source_faces" in data:
        source_vertices = _select_source(data["source_vertices"], source_index)
        source_faces = _select_source(data["source_faces"], source_index).astype(np.int32)
        viewer.scene.add(
            Meshes(
                source_vertices,
                source_faces,
                name="source scan",
                color=(0.55, 0.55, 0.55, 0.30),
                cast_shadow=False,
                draw_edges=False,
            )
        )

    viewer.scene.add(
        Meshes(
            vertices,
            faces,
            name="SMPL-X fitting trajectory",
            color=(1.0, 0.05, 0.65, 0.92),
            cast_shadow=False,
            draw_edges=False,
        )
    )

    if show_joints and Spheres is not None:
        joints = _select_batch(data["joints"], batch_index)
        viewer.scene.add(
            Spheres(
                joints,
                radius=0.012,
                color=(1.0, 0.05, 0.65, 1.0),
                name="fitted joints",
            )
        )
        target_joints = _select_batch(data["target_joints"], batch_index)
        viewer.scene.add(
            Spheres(
                target_joints,
                radius=0.01,
                color=(0.0, 0.75, 1.0, 0.80),
                name="target keypoints",
            )
        )

    if (
        camera_json is not None
        and initial_camera is not None
        and initial_camera.lower() != "none"
    ):
        camera_id = set_initial_camera_view(
            viewer,
            InitialCameraConfig(
                camera_json=camera_json,
                camera_id=None if initial_camera.lower() == "auto" else initial_camera,
                camera_scale=camera_scale,
            ),
            existing_cameras=added_cameras,
        )
        print(f"Initialized AITviewer view from camera {camera_id}.")

    print(
        f"Loaded {recording_path} with {vertices.shape[0]} snapshots. "
        "Use AITviewer's timeline controls to scrub the fit."
    )
    viewer.run()


def _bind_legacy_render_loop(viewer) -> None:
    """Bind AITviewer 1.13 render callbacks when moderngl-window misses them."""
    window = getattr(viewer, "window", None)
    render_func = getattr(viewer, "render", None) or getattr(viewer, "on_render", None)
    if window is not None and render_func is not None:
        window.render_func = render_func


def _load_metadata(data) -> dict[str, Any]:
    if "metadata_json" not in data:
        return {}
    raw = data["metadata_json"]
    if hasattr(raw, "item"):
        raw = raw.item()
    try:
        metadata = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return metadata if isinstance(metadata, dict) else {}


def _resolve_optional_path(value: Path | None, fallback: Any) -> Path | None:
    if value is not None:
        return value
    if fallback is None:
        return None
    fallback = str(fallback)
    return Path(fallback) if fallback else None


def _metadata_frame_ids(metadata: dict[str, Any]) -> tuple[int, ...] | None:
    values = metadata.get("mesh_frame_ids")
    if not values:
        return None
    return tuple(int(value) for value in values)


def _to_float32(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _select_batch(array: np.ndarray, batch_index: int) -> np.ndarray:
    if array.ndim == 3:
        return np.asarray(array, dtype=np.float32)
    if array.ndim != 4:
        raise ValueError(f"Expected recorded array with shape (T, B, ..., C), got {array.shape}.")
    if batch_index < 0 or batch_index >= array.shape[1]:
        raise ValueError(f"batch_index {batch_index} out of range for batch size {array.shape[1]}.")
    return np.asarray(array[:, batch_index], dtype=np.float32)


def _select_source(array: np.ndarray, source_index: int) -> np.ndarray:
    if array.ndim == 2:
        return np.asarray(array)
    if array.ndim != 3:
        raise ValueError(f"Unexpected source mesh array shape: {array.shape}")
    if source_index < 0 or source_index >= array.shape[0]:
        raise ValueError(f"source_index {source_index} out of range for {array.shape[0]} sources.")
    return np.asarray(array[source_index])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="Path to a recorded .npz file.")
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--source-index", type=int, default=0)
    parser.add_argument("--hide-source", action="store_true")
    parser.add_argument("--hide-joints", action="store_true")
    parser.add_argument("--hide-cameras", action="store_true")
    parser.add_argument("--camera-json", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--frame-ids", default=None)
    parser.add_argument("--cameras", default=None)
    parser.add_argument("--max-cameras", type=int, default=4)
    parser.add_argument(
        "--calibration-scale",
        "--camera-scale",
        dest="camera_scale",
        type=float,
        default=0.001,
        help=(
            "Scale camera translations into the mesh coordinate system. This must match "
            "input.scale_to_meters; use --billboard-distance for visual size."
        ),
    )
    parser.add_argument(
        "--initial-camera",
        default="auto",
        help="'auto' uses the first selected camera, 'none' disables camera init, or pass a camera id.",
    )
    parser.add_argument("--billboard-distance", type=float, default=2.0)
    parser.add_argument("--billboard-alpha", type=float, default=0.55)
    parser.add_argument("--image-extensions", default=".png,.jpg,.jpeg")
    parser.add_argument("--floor", action="store_true")
    parser.add_argument("--window-type", default=None)
    parser.add_argument("--shadows", action="store_true")
    parser.add_argument("--znear", type=float, default=0.05)
    parser.add_argument("--zfar", type=float, default=50.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    launch_recording_viewer(
        args.recording,
        batch_index=args.batch_index,
        source_index=args.source_index,
        show_source=not args.hide_source,
        show_joints=not args.hide_joints,
        show_cameras=not args.hide_cameras,
        camera_json=args.camera_json,
        image_root=args.image_root,
        frame_ids=parse_frame_ids(args.frame_ids) if args.frame_ids else None,
        camera_ids=parse_camera_ids(args.cameras),
        max_cameras=args.max_cameras,
        camera_scale=args.camera_scale,
        initial_camera=args.initial_camera,
        billboard_distance=args.billboard_distance,
        billboard_alpha=args.billboard_alpha,
        image_extensions=tuple(
            item.strip() for item in args.image_extensions.split(",") if item.strip()
        ),
        floor_enabled=args.floor,
        render_config=ViewerRenderConfig(
            window_type=args.window_type,
            shadows_enabled=args.shadows,
            znear=args.znear,
            zfar=args.zfar,
        ),
    )


if __name__ == "__main__":
    main()
