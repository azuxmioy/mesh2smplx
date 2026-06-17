"""Shared data contracts for real and virtual input sources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class CameraModel:
    camera_id: str
    intrinsics: np.ndarray
    extrinsics: np.ndarray
    image_size: tuple[int, int]
    dist_coeffs: np.ndarray | None = None


@dataclass(frozen=True)
class FrameObservation:
    frame_id: int
    image_paths: dict[str, Path]
    mask_paths: dict[str, Path] | None = None
    mesh_path: Path | None = None
    texture_path: Path | None = None


@dataclass(frozen=True)
class ObservationBundle:
    cameras: dict[str, CameraModel]
    frames: list[FrameObservation]

    @property
    def frame_ids(self) -> list[int]:
        return [frame.frame_id for frame in self.frames]


class DataSource(Protocol):
    def load(self, frames: list[int] | None = None) -> ObservationBundle:
        """Load observations for requested frames."""

