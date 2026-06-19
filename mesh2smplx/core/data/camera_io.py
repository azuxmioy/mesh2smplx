"""Camera calibration loading helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .interfaces import CameraModel


def resolve_camera_json(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        candidates = [
            path / "cameras.json",
            path / "rgb_cameras.json",
            path / "calibration.json",
            path / "camera.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        json_files = sorted(path.glob("*.json"))
        if len(json_files) == 1:
            return json_files[0]
    raise FileNotFoundError(
        f"Camera calibration not found: {path}. Expected a JSON file or a directory "
        "containing cameras.json, rgb_cameras.json, calibration.json, or one JSON file."
    )


def load_camera_models(
    camera_json: Path,
    images_root: Path | None = None,
    image_glob: str = "*.png",
) -> dict[str, CameraModel]:
    payload = json.loads(camera_json.read_text(encoding="utf-8"))
    camera_payload = payload.get("cameras", payload) if isinstance(payload, dict) else payload
    if not isinstance(camera_payload, dict):
        raise ValueError("Camera calibration JSON must be a dict of camera_id -> camera data.")

    cameras = {}
    for camera_id, data in camera_payload.items():
        if not isinstance(data, dict):
            raise ValueError(f"Camera {camera_id} must be an object.")
        intrinsics = np.asarray(data["intrinsics"], dtype=np.float32)
        extrinsics = np.asarray(data["extrinsics"], dtype=np.float32)
        if extrinsics.shape == (4, 4):
            extrinsics = extrinsics[:3]
        if intrinsics.shape != (3, 3) or extrinsics.shape != (3, 4):
            raise ValueError(
                f"Camera {camera_id} expects intrinsics 3x3 and extrinsics 3x4 or 4x4."
            )
        cameras[str(camera_id)] = CameraModel(
            camera_id=str(camera_id),
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            image_size=_camera_image_size(data, images_root, str(camera_id), image_glob),
            dist_coeffs=(
                np.asarray(data["dist_coeffs"], dtype=np.float32)
                if "dist_coeffs" in data
                else None
            ),
        )
    return cameras


def camera_image_dir(images_root: Path, camera_id: str) -> Path:
    camera_dir = images_root / camera_id
    if camera_dir.exists():
        return camera_dir
    return images_root


def _camera_image_size(
    data: dict,
    images_root: Path | None,
    camera_id: str,
    image_glob: str,
) -> tuple[int, int]:
    if "image_size" in data:
        height, width = data["image_size"]
        return int(height), int(width)
    if "shape" in data:
        height, width = data["shape"]
        return int(height), int(width)
    if "height" in data and "width" in data:
        return int(data["height"]), int(data["width"])
    if images_root is None:
        raise ValueError(
            f"Camera {camera_id} is missing image_size. Add [height, width] when "
            "using calibration without existing images."
        )

    image_dir = camera_image_dir(images_root, camera_id)
    first_image = next(iter(sorted(image_dir.glob(image_glob))), None)
    if first_image is None:
        raise FileNotFoundError(f"No images matched {image_dir / image_glob}")
    with Image.open(first_image) as image:
        width, height = image.size
    return int(height), int(width)
