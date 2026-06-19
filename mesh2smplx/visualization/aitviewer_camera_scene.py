"""AITviewer scene helpers for calibrated cameras and image billboards."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraImageOverlayConfig:
    camera_json: Path
    image_root: Path
    frame_ids: tuple[int, ...]
    camera_ids: tuple[str, ...] | None = None
    max_cameras: int = 4
    camera_scale: float = 0.001
    billboard_distance: float = 2.0
    billboard_alpha: float = 0.55
    image_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class InitialCameraConfig:
    camera_json: Path
    camera_id: str | None = None
    camera_scale: float = 0.001
    target_distance: float = 3.0


@dataclass(frozen=True)
class ViewerRenderConfig:
    window_type: str | None = None
    shadows_enabled: bool = False
    znear: float = 0.05
    zfar: float = 50.0


def parse_camera_ids(value: str | None) -> tuple[str, ...] | None:
    if value is None or not value.strip():
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_frame_ids(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def format_camera_ids(camera_ids: tuple[str, ...] | None) -> str | None:
    if camera_ids is None:
        return None
    return ",".join(camera_ids)


def format_frame_ids(frame_ids: tuple[int, ...]) -> str:
    return ",".join(str(frame_id) for frame_id in frame_ids)


def add_camera_image_overlays(viewer, config: CameraImageOverlayConfig) -> dict[str, Any]:
    """Add calibrated OpenCV cameras, frustums, and image billboards to a viewer."""
    from aitviewer.renderables.billboard import Billboard

    cameras = json.loads(config.camera_json.read_text(encoding="utf-8"))
    camera_ids = _select_camera_ids(cameras, config.camera_ids, config.max_cameras)
    added: dict[str, Any] = {}

    for camera_id in camera_ids:
        camera_data = cameras.get(str(camera_id))
        if camera_data is None:
            print(f"Warning: camera id {camera_id} not found in {config.camera_json}")
            continue

        camera, rows, cols = _opencv_camera_from_data(
            viewer,
            camera_id=str(camera_id),
            camera_data=camera_data,
            camera_scale=config.camera_scale,
            name=f"camera {camera_id}",
        )
        viewer.scene.add(camera)
        camera.show_frustum(int(cols), int(rows), config.billboard_distance)
        added[str(camera_id)] = camera

        image_paths, missing = _image_paths_for_camera(config, str(camera_id))
        if missing:
            preview = ", ".join(str(frame_id) for frame_id in missing[:4])
            print(f"Warning: camera {camera_id} missing images for frame ids: {preview}")
        if not image_paths:
            continue

        billboard = Billboard.from_camera_and_distance(
            camera,
            config.billboard_distance,
            int(cols),
            int(rows),
            [str(path) for path in image_paths],
            name=f"camera {camera_id} images",
        )
        billboard.texture_alpha = config.billboard_alpha
        viewer.scene.add(billboard)

    return added


def launch_camera_scene(
    config: CameraImageOverlayConfig,
    server_port: int = 8417,
    initial_camera: InitialCameraConfig | None = None,
    server_enabled: bool = True,
    render_config: ViewerRenderConfig | None = None,
) -> None:
    """Launch an AITviewer server scene with calibrated camera image overlays."""
    from aitviewer.configuration import CONFIG as C
    from aitviewer.viewer import Viewer

    update_viewer_config(C, server_enabled, server_port, render_config)
    viewer = Viewer()
    viewer.scene.floor.enabled = False
    added = add_camera_image_overlays(viewer, config)
    print(f"Added {len(added)} calibrated cameras with image overlays.")
    if initial_camera is not None:
        camera_id = set_initial_camera_view(viewer, initial_camera, existing_cameras=added)
        print(f"Initialized AITviewer view from camera {camera_id}.")
    if server_enabled:
        print(f"Started AITviewer remote server on port {server_port}.")
    else:
        print("Started AITviewer without remote server.")
    viewer.run()


def launch_empty_server_scene(
    server_port: int = 8417,
    initial_camera: InitialCameraConfig | None = None,
    server_enabled: bool = True,
    render_config: ViewerRenderConfig | None = None,
) -> None:
    """Launch an empty AITviewer scene that accepts remote streamed nodes."""
    from aitviewer.configuration import CONFIG as C
    from aitviewer.viewer import Viewer

    update_viewer_config(C, server_enabled, server_port, render_config)
    viewer = Viewer()
    viewer.scene.floor.enabled = False
    if initial_camera is not None:
        camera_id = set_initial_camera_view(viewer, initial_camera)
        print(f"Initialized AITviewer view from camera {camera_id}.")
    if server_enabled:
        print(f"Started AITviewer remote server on port {server_port}.")
    else:
        print("Started AITviewer without remote server.")
    viewer.run()


def update_viewer_config(
    config,
    server_enabled: bool,
    server_port: int,
    render_config: ViewerRenderConfig | None,
) -> None:
    render_config = render_config or ViewerRenderConfig()
    values = {
        "server_enabled": server_enabled,
        "server_port": server_port,
        "shadows_enabled": render_config.shadows_enabled,
        "znear": render_config.znear,
        "zfar": render_config.zfar,
    }
    if render_config.window_type:
        values["window_type"] = render_config.window_type
    config.update_conf(values)


def set_initial_camera_view(
    viewer,
    config: InitialCameraConfig,
    existing_cameras: dict[str, Any] | None = None,
) -> str:
    """Set the interactive viewer camera from a calibrated OpenCV camera."""
    cameras = json.loads(config.camera_json.read_text(encoding="utf-8"))
    camera_id = _select_initial_camera_id(cameras, config.camera_id)

    camera = None if existing_cameras is None else existing_cameras.get(camera_id)
    if camera is None:
        camera_data = cameras[camera_id]
        camera, _, _ = _opencv_camera_from_data(
            viewer,
            camera_id=camera_id,
            camera_data=camera_data,
            camera_scale=config.camera_scale,
            name=f"initial camera {camera_id}",
        )

    viewport = viewer.viewports[0]
    if not hasattr(viewport.camera, "target"):
        viewport.reset_camera()
    interactive_camera = viewport.camera
    interactive_camera.position = camera.position
    interactive_camera.target = _interactive_target_from_opencv_camera(
        camera,
        fallback_distance=config.target_distance,
    )
    interactive_camera.up = camera.up
    interactive_camera.fov = _fov_from_opencv_camera(camera)
    interactive_camera.update_matrices(*viewport.extents[2:])
    viewer.scene.camera = interactive_camera
    viewer.auto_set_camera_target = False
    return camera_id


def _select_camera_ids(
    cameras: dict[str, object],
    requested: tuple[str, ...] | None,
    max_cameras: int,
) -> tuple[str, ...]:
    if requested is not None:
        return requested

    def sort_key(value: str) -> tuple[int, int | str]:
        return (0, int(value)) if value.isdigit() else (1, value)

    return tuple(sorted(cameras.keys(), key=sort_key)[: max(1, max_cameras)])


def _select_initial_camera_id(cameras: dict[str, object], camera_id: str | None) -> str:
    if camera_id is not None:
        camera_id = str(camera_id)
        if camera_id not in cameras:
            raise ValueError(f"Initial AITviewer camera {camera_id!r} is not in the calibration JSON.")
        return camera_id
    selected = _select_camera_ids(cameras, requested=None, max_cameras=1)
    if not selected:
        raise ValueError("Calibration JSON does not contain any cameras.")
    return selected[0]


def _opencv_camera_from_data(
    viewer,
    *,
    camera_id: str,
    camera_data: object,
    camera_scale: float,
    name: str,
):
    from aitviewer.scene.camera import OpenCVCamera

    if not isinstance(camera_data, dict):
        raise ValueError(f"Camera {camera_id} must be a JSON object.")
    intrinsics = np.asarray(camera_data["intrinsics"], dtype=np.float64)
    extrinsics = np.asarray(camera_data["extrinsics"], dtype=np.float64).copy()
    extrinsics[:, 3] *= camera_scale
    rows, cols = camera_data["shape"]
    camera = OpenCVCamera(
        intrinsics,
        extrinsics,
        cols=int(cols),
        rows=int(rows),
        dist_coeffs=np.asarray(camera_data.get("dist_coeffs"), dtype=np.float64)
        if camera_data.get("dist_coeffs") is not None
        else None,
        viewer=viewer,
        name=name,
    )
    return camera, int(rows), int(cols)


def _interactive_target_from_opencv_camera(camera, fallback_distance: float) -> np.ndarray:
    position = np.asarray(camera.position, dtype=np.float64)
    forward = _normalize(np.asarray(camera.forward, dtype=np.float64))
    distance_to_origin = float(np.dot(-position, forward))
    if distance_to_origin <= 0.25:
        distance_to_origin = max(0.25, fallback_distance)
    return position + forward * distance_to_origin


def _fov_from_opencv_camera(camera) -> float:
    intrinsics = camera.current_K
    return float(np.rad2deg(2.0 * np.arctan(float(camera.rows) / (2.0 * intrinsics[1, 1]))))


def _normalize(value: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(value)
    if norm <= 1e-8:
        return value
    return value / norm


def _image_paths_for_camera(
    config: CameraImageOverlayConfig,
    camera_id: str,
) -> tuple[list[Path], list[int]]:
    image_paths = []
    missing = []
    for frame_id in config.frame_ids:
        path = _find_image_path(config.image_root / camera_id, frame_id, config.image_extensions)
        if path is None:
            missing.append(frame_id)
        else:
            image_paths.append(path)
    return image_paths, missing


def _find_image_path(camera_dir: Path, frame_id: int, extensions: tuple[str, ...]) -> Path | None:
    for extension in extensions:
        path = camera_dir / f"{frame_id:06d}{extension}"
        if path.exists():
            return path
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-json", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--frame-ids", default="")
    parser.add_argument("--cameras", default=None)
    parser.add_argument("--initial-camera-json", type=Path)
    parser.add_argument("--initial-camera-id", default=None)
    parser.add_argument("--max-cameras", type=int, default=4)
    parser.add_argument("--camera-scale", type=float, default=0.001)
    parser.add_argument("--billboard-distance", type=float, default=2.0)
    parser.add_argument("--billboard-alpha", type=float, default=0.55)
    parser.add_argument("--image-extensions", default=".png,.jpg,.jpeg")
    parser.add_argument("--server-port", type=int, default=8417)
    parser.add_argument("--no-server", action="store_true", help="Launch AITviewer without websocket.")
    parser.add_argument("--window-type", default=None)
    parser.add_argument("--shadows", action="store_true", help="Enable AITviewer shadows.")
    parser.add_argument("--znear", type=float, default=0.05)
    parser.add_argument("--zfar", type=float, default=50.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    initial_camera_json = args.initial_camera_json or args.camera_json
    initial_camera = (
        InitialCameraConfig(
            camera_json=initial_camera_json,
            camera_id=args.initial_camera_id,
            camera_scale=args.camera_scale,
        )
        if initial_camera_json is not None
        else None
    )
    if args.initial_camera_id is not None and initial_camera is None:
        raise ValueError("--initial-camera-id requires --initial-camera-json or --camera-json.")
    render_config = ViewerRenderConfig(
        window_type=args.window_type,
        shadows_enabled=args.shadows,
        znear=args.znear,
        zfar=args.zfar,
    )

    if args.camera_json is None or args.image_root is None:
        launch_empty_server_scene(
            server_port=args.server_port,
            initial_camera=initial_camera,
            server_enabled=not args.no_server,
            render_config=render_config,
        )
        return
    if not args.frame_ids:
        raise ValueError("--frame-ids is required when camera overlays are enabled.")
    config = CameraImageOverlayConfig(
        camera_json=args.camera_json,
        image_root=args.image_root,
        frame_ids=parse_frame_ids(args.frame_ids),
        camera_ids=parse_camera_ids(args.cameras),
        max_cameras=args.max_cameras,
        camera_scale=args.camera_scale,
        billboard_distance=args.billboard_distance,
        billboard_alpha=args.billboard_alpha,
        image_extensions=tuple(
            item.strip() for item in args.image_extensions.split(",") if item.strip()
        ),
    )
    launch_camera_scene(
        config,
        server_port=args.server_port,
        initial_camera=initial_camera,
        server_enabled=not args.no_server,
        render_config=render_config,
    )


if __name__ == "__main__":
    main()
