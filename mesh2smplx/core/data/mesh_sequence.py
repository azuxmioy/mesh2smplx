"""Textured mesh sequence discovery."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..config import InputConfig

MESH_EXTENSIONS = {".obj", ".ply", ".stl", ".glb", ".gltf", ".off"}


@dataclass(frozen=True)
class MeshFrame:
    frame_id: int
    mesh_path: Path
    texture_path: Path | None = None


def discover_mesh_sequence(config: InputConfig, frames: list[int] | None = None) -> list[MeshFrame]:
    if config.meshes is None:
        raise ValueError("textured_mesh mode requires input.meshes")

    mesh_paths = _discover_mesh_paths(config)
    if not mesh_paths:
        raise FileNotFoundError(f"No meshes matched {config.meshes / config.mesh_glob}")

    frame_id_re = re.compile(config.frame_id_regex)
    texture_by_frame_id = _index_textures(config, frame_id_re)

    discovered: list[MeshFrame] = []
    seen_frame_ids: dict[int, Path] = {}
    for index, mesh_path in enumerate(mesh_paths):
        frame_id = _parse_path_frame_id(mesh_path, config.meshes, frame_id_re, fallback=index)
        if frame_id in seen_frame_ids:
            raise ValueError(
                "Duplicate mesh frame id "
                f"{frame_id}: {seen_frame_ids[frame_id]} and {mesh_path}. "
                "Adjust input.mesh_glob or input.frame_id_regex so each frame has one mesh."
            )
        seen_frame_ids[frame_id] = mesh_path
        texture_path = texture_by_frame_id.get(frame_id)
        discovered.append(MeshFrame(frame_id=frame_id, mesh_path=mesh_path, texture_path=texture_path))

    discovered.sort(key=lambda frame: frame.frame_id)

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
        frame_id = _parse_path_frame_id(path, config.textures, frame_id_re, fallback=index)
        indexed[frame_id] = path
    return indexed


def _discover_mesh_paths(config: InputConfig) -> list[Path]:
    assert config.meshes is not None
    paths = [
        path
        for path in config.meshes.glob(config.mesh_glob)
        if path.is_file() and path.suffix.lower() in MESH_EXTENSIONS
    ]
    return sorted(paths)


def _parse_path_frame_id(path: Path, root: Path, frame_id_re: re.Pattern[str], fallback: int) -> int:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    candidates = [
        path.stem,
        path.parent.name,
        relative.with_suffix("").as_posix(),
        relative.as_posix(),
    ]
    for candidate in candidates:
        frame_id = _parse_frame_id(candidate, frame_id_re)
        if frame_id is not None:
            return frame_id
    return fallback


def _parse_frame_id(
    text: str,
    frame_id_re: re.Pattern[str],
    fallback: int | None = None,
) -> int | None:
    match = frame_id_re.match(text)
    if match:
        return int(match.group(1))
    return fallback
