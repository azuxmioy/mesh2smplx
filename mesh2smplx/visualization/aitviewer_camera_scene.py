"""AITviewer scene helpers for calibrated cameras and image billboards."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

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


def add_camera_image_overlays(viewer, config: CameraImageOverlayConfig) -> int:
    """Add calibrated OpenCV cameras, frustums, and image billboards to a viewer."""
    from aitviewer.renderables.billboard import Billboard
    from aitviewer.scene.camera import OpenCVCamera

    cameras = json.loads(config.camera_json.read_text(encoding="utf-8"))
    camera_ids = _select_camera_ids(cameras, config.camera_ids, config.max_cameras)
    added = 0

    for camera_id in camera_ids:
        camera_data = cameras.get(str(camera_id))
        if camera_data is None:
            print(f"Warning: camera id {camera_id} not found in {config.camera_json}")
            continue

        intrinsics = np.asarray(camera_data["intrinsics"], dtype=np.float64)
        extrinsics = np.asarray(camera_data["extrinsics"], dtype=np.float64).copy()
        extrinsics[:, 3] *= config.camera_scale
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
            name=f"camera {camera_id}",
        )
        viewer.scene.add(camera)
        camera.show_frustum(int(cols), int(rows), config.billboard_distance)
        added += 1

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


def launch_camera_scene(config: CameraImageOverlayConfig, server_port: int = 8417) -> None:
    """Launch an AITviewer server scene with calibrated camera image overlays."""
    from aitviewer.configuration import CONFIG as C
    from aitviewer.viewer import Viewer

    C.update_conf({"server_enabled": True, "server_port": server_port})
    viewer = Viewer()
    viewer.scene.floor.enabled = False
    added = add_camera_image_overlays(viewer, config)
    print(f"Added {added} calibrated cameras with image overlays.")
    viewer.run()


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
    parser.add_argument("--camera-json", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--frame-ids", required=True)
    parser.add_argument("--cameras", default=None)
    parser.add_argument("--max-cameras", type=int, default=4)
    parser.add_argument("--camera-scale", type=float, default=0.001)
    parser.add_argument("--billboard-distance", type=float, default=2.0)
    parser.add_argument("--billboard-alpha", type=float, default=0.55)
    parser.add_argument("--image-extensions", default=".png,.jpg,.jpeg")
    parser.add_argument("--server-port", type=int, default=8417)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
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
    launch_camera_scene(config, server_port=args.server_port)


if __name__ == "__main__":
    main()
