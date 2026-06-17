"""Textured mesh source rendered from virtual cameras."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import InputConfig, VirtualCameraConfig
from .interfaces import ObservationBundle


@dataclass
class TexturedMeshSource:
    input_config: InputConfig
    camera_config: VirtualCameraConfig

    def load(self, frames: list[int] | None = None) -> ObservationBundle:
        from .mesh_sequence import discover_mesh_sequence
        from ..render.virtual_cameras import render_textured_mesh_sequence

        mesh_frames = discover_mesh_sequence(self.input_config, frames=frames)
        return render_textured_mesh_sequence(
            input_config=self.input_config,
            camera_config=self.camera_config,
            mesh_frames=mesh_frames,
        )
