"""Keypoint providers: precomputed JSON, external command, or vendored OpenPose-135."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from mesh2smplx.core.config import KeypointConfig
from mesh2smplx.core.data.interfaces import ObservationBundle
from .format import keypoint_json_path


class KeypointProvider(Protocol):
    def run(self, observations: ObservationBundle) -> None:
        """Produce or validate 2D keypoints for an observation bundle."""


@dataclass
class PrecomputedKeypoints:
    config: KeypointConfig

    def run(self, observations: ObservationBundle) -> None:
        if self.config.path is None:
            raise ValueError("precomputed keypoints require keypoints.path")
        if not self.config.path.exists():
            raise FileNotFoundError(self.config.path)


@dataclass
class ExternalCommandKeypoints:
    config: KeypointConfig

    def run(self, observations: ObservationBundle) -> None:
        if not self.config.command:
            raise ValueError("external_command keypoints require keypoints.command")
        raise NotImplementedError(
            "external_command keypoint execution is not wired yet. Run your detector "
            "separately and use provider: precomputed."
        )


def _resolve_device(spec: str | None) -> str:
    if spec is not None and spec != "auto":
        return spec
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@dataclass
class OpenPose135Keypoints:
    """Run the vendored OpenPose-135 detector over an observation bundle.

    Writes one CMU OpenPose JSON per camera and frame to::

        <keypoints.output_dir>/<camera_id>/<frame_id:06d>_keypoints.json

    which is exactly the layout :func:`format.load_frame_keypoints`
    reads back for triangulation. The CMU OpenPose model weights are
    non-commercial; they are downloaded at runtime (HF mirror / local cache),
    never shipped with the package.
    """

    config: KeypointConfig

    def _build_detector(self):
        from . import OpenPose135Detector
        from .weights import resolve_weights

        kinds = ["body25"]
        if self.config.enable_hand:
            kinds.append("hand")
        if self.config.enable_face:
            kinds.append("face")

        cache_dir = self.config.weights_dir or os.environ.get("OPENPOSE135_CACHE_DIR")
        weight_paths = resolve_weights(
            repo_id=self.config.hf_repo,
            cache_dir=str(cache_dir) if cache_dir else None,
            kinds=kinds,
        )
        return OpenPose135Detector(
            device=_resolve_device(self.config.device),
            weight_paths=weight_paths,
            enable_hand=self.config.enable_hand,
            enable_face=self.config.enable_face,
        )

    def run(self, observations: ObservationBundle) -> None:
        if self.config.output_dir is None:
            raise ValueError("openpose135 keypoints require keypoints.output_dir")

        from .runtime import process_image

        output_dir = self.config.output_dir
        render_pose = 2 if self.config.render_overlays else 0
        detector = self._build_detector()

        for frame in observations.frames:
            for camera_id, image_path in frame.image_paths.items():
                json_path = keypoint_json_path(output_dir, camera_id, frame.frame_id)
                overlay_path = (
                    json_path.with_name(f"{frame.frame_id:06d}_overlay.png")
                    if self.config.render_overlays
                    else None
                )
                if not self.config.overwrite and json_path.exists():
                    continue
                if not image_path.exists():
                    raise FileNotFoundError(
                        f"OpenPose input image missing for camera {camera_id}, "
                        f"frame {frame.frame_id}: {image_path}"
                    )
                json_path.parent.mkdir(parents=True, exist_ok=True)
                process_image(
                    detector,
                    image_path,
                    write_json=json_path,
                    write_image=overlay_path,
                    render_pose=render_pose,
                    number_people_max=self.config.number_people_max,
                )


def build_keypoint_provider(config: KeypointConfig) -> KeypointProvider:
    if config.provider == "precomputed":
        return PrecomputedKeypoints(config)
    if config.provider == "external_command":
        return ExternalCommandKeypoints(config)
    if config.provider == "openpose135":
        return OpenPose135Keypoints(config)
    raise ValueError(f"Unsupported keypoint provider: {config.provider}")
