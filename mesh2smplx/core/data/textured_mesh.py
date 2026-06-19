"""Textured mesh source from calibrated images or virtual cameras."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import InputConfig, VirtualCameraConfig
from .camera_io import camera_image_dir, load_camera_models, resolve_camera_json
from .interfaces import FrameObservation, ObservationBundle


@dataclass
class TexturedMeshSource:
    input_config: InputConfig
    camera_config: VirtualCameraConfig | None

    def load(self, frames: list[int] | None = None) -> ObservationBundle:
        from .mesh_sequence import discover_mesh_sequence
        from ..render.virtual_cameras import render_textured_mesh_sequence

        mesh_frames = discover_mesh_sequence(self.input_config, frames=frames)
        if self.input_config.images is not None:
            return load_calibrated_image_sequence(self.input_config, mesh_frames)
        if self.camera_config is None:
            raise ValueError("textured_mesh mode without input.images requires virtual_cameras config")
        return render_textured_mesh_sequence(
            input_config=self.input_config,
            camera_config=self.camera_config,
            mesh_frames=mesh_frames,
        )


def load_calibrated_image_sequence(input_config: InputConfig, mesh_frames) -> ObservationBundle:
    if input_config.images is None:
        raise ValueError("calibrated image loading requires input.images")
    if input_config.cameras is None:
        raise ValueError(
            "input.images requires camera calibration. Set input.cameras or "
            "input.calibration to a camera JSON file or calibration directory."
        )

    camera_json = resolve_camera_json(input_config.cameras)
    images_root = input_config.images
    cameras = load_camera_models(camera_json, images_root, input_config.image_glob)
    image_index = {
        camera_id: _index_camera_images(images_root, camera_id, input_config)
        for camera_id in cameras
    }

    observations = []
    for mesh_frame in mesh_frames:
        image_paths = {}
        for camera_id in cameras:
            try:
                image_paths[camera_id] = image_index[camera_id][mesh_frame.frame_id]
            except KeyError as exc:
                raise FileNotFoundError(
                    f"Missing image for camera {camera_id}, frame {mesh_frame.frame_id} "
                    f"under {images_root}"
                ) from exc
        observations.append(
            FrameObservation(
                frame_id=mesh_frame.frame_id,
                image_paths=image_paths,
                mesh_path=mesh_frame.mesh_path,
                texture_path=mesh_frame.texture_path,
            )
        )

    return ObservationBundle(cameras=cameras, frames=observations)


def _index_camera_images(
    images_root: Path,
    camera_id: str,
    input_config: InputConfig,
) -> dict[int, Path]:
    import re

    from .mesh_sequence import _parse_frame_id

    image_dir = camera_image_dir(images_root, camera_id)
    frame_id_re = re.compile(input_config.frame_id_regex)
    indexed = {}
    for index, path in enumerate(sorted(image_dir.glob(input_config.image_glob))):
        frame_id = _parse_frame_id(path.stem, frame_id_re, fallback=index)
        indexed[frame_id] = path
    if not indexed:
        raise FileNotFoundError(f"No images matched {image_dir / input_config.image_glob}")
    return indexed
