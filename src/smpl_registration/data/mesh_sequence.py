"""Textured mesh sequence discovery."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..config import InputConfig


@dataclass(frozen=True)
class MeshFrame:
    frame_id: int
    mesh_path: Path
    texture_path: Path | None = None


def discover_mesh_sequence(config: InputConfig, frames: list[int] | None = None) -> list[MeshFrame]:
    if config.meshes is None:
        raise ValueError("textured_mesh mode requires input.meshes")

    mesh_paths = sorted(config.meshes.glob(config.mesh_glob))
    if not mesh_paths:
        raise FileNotFoundError(f"No meshes matched {config.meshes / config.mesh_glob}")

    frame_id_re = re.compile(config.frame_id_regex)
    texture_by_frame_id = _index_textures(config, frame_id_re)

    discovered: list[MeshFrame] = []
    for index, mesh_path in enumerate(mesh_paths):
        frame_id = _parse_frame_id(mesh_path.stem, frame_id_re, fallback=index)
        texture_path = texture_by_frame_id.get(frame_id)
        discovered.append(MeshFrame(frame_id=frame_id, mesh_path=mesh_path, texture_path=texture_path))

    if frames is None:
        return discovered

    requested = set(frames)
    filtered = [frame for frame in discovered if frame.frame_id in requested]
    missing = sorted(requested.difference(frame.frame_id for frame in filtered))
    if missing:
        raise ValueError(f"Requested frames not found in mesh sequence: {missing}")
    return filtered


def _index_textures(config: InputConfig, frame_id_re: re.Pattern[str]) -> dict[int, Path]:
    if config.textures is None or config.texture_glob is None:
        return {}
    indexed = {}
    for index, path in enumerate(sorted(config.textures.glob(config.texture_glob))):
        frame_id = _parse_frame_id(path.stem, frame_id_re, fallback=index)
        indexed[frame_id] = path
    return indexed


def _parse_frame_id(stem: str, frame_id_re: re.Pattern[str], fallback: int) -> int:
    match = frame_id_re.match(stem)
    if match:
        return int(match.group(1))
    return fallback
