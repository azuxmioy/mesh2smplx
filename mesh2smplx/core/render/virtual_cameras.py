"""Virtual-camera observation records for textured mesh sequences.

Kaolin is intentionally optional and imported only inside rendering helpers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import InputConfig, VirtualCameraConfig
from ..data.mesh_sequence import MeshFrame
from ..data.interfaces import CameraModel, FrameObservation, ObservationBundle


def require_kaolin():
    try:
        import kaolin
    except ImportError as exc:
        raise RuntimeError(
            "Virtual-camera rendering requires the optional render extra. "
            "Install `smpl-registration[render]` in the final package."
        ) from exc
    return kaolin


def build_orbit_camera_ids(count: int) -> list[str]:
    return [f"virtual_{idx:03d}" for idx in range(count)]


def placeholder_virtual_cameras(config: VirtualCameraConfig) -> dict[str, CameraModel]:
    """Create placeholder camera records until Kaolin rendering is wired."""
    cameras: dict[str, CameraModel] = {}
    for camera_id in build_orbit_camera_ids(config.count):
        cameras[camera_id] = CameraModel(
            camera_id=camera_id,
            intrinsics=np.array(
                [
                    [config.focal_length, 0.0, config.width / 2.0],
                    [0.0, config.focal_length, config.height / 2.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            extrinsics=np.eye(3, 4, dtype=np.float32),
            image_size=(config.height, config.width),
        )
    return cameras


def render_textured_mesh_sequence(
    input_config: InputConfig,
    camera_config: VirtualCameraConfig,
    mesh_frames: list[MeshFrame],
) -> ObservationBundle:
    """Build virtual-camera observations for textured meshes.

    The returned bundle contains the camera records and output paths expected by
    downstream keypoint detection. Rendering backends should write images and
    masks into the configured output directory before keypoints are generated.
    """
    if input_config.meshes is None:
        raise ValueError("textured_mesh mode requires input.meshes")

    cameras = placeholder_virtual_cameras(camera_config)
    observations = []
    image_root = camera_config.output_dir / "images"
    mask_root = camera_config.output_dir / "masks"
    for mesh_frame in mesh_frames:
        frame_id = mesh_frame.frame_id
        observations.append(
            FrameObservation(
                frame_id=frame_id,
                image_paths={
                    camera_id: image_root / camera_id / f"{frame_id:06d}.png"
                    for camera_id in cameras
                },
                mask_paths={
                    camera_id: mask_root / camera_id / f"{frame_id:06d}.png"
                    for camera_id in cameras
                }
                if camera_config.render_masks
                else None,
                mesh_path=mesh_frame.mesh_path,
                texture_path=mesh_frame.texture_path,
            )
        )
    return ObservationBundle(cameras=cameras, frames=observations)
