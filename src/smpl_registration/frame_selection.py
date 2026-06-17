"""Frame selection utilities shared by CLI and data sources."""

from __future__ import annotations

import re
from pathlib import Path


def parse_frame_range(spec: str | None) -> list[int] | None:
    """Parse comma-separated frame ids and inclusive ranges."""
    if spec is None or spec == "":
        return None

    frames: list[int] = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for part in spec.split(","):
        part = part.strip()
        match = range_re.match(part)
        if match:
            start = int(match.group(1))
            stop = int(match.group(2))
            if stop < start:
                raise ValueError(f"Invalid descending frame range: {part}")
            frames.extend(range(start, stop + 1))
        else:
            frames.append(int(part))
    return frames


def frames_from_atlas(atlas_dir: Path, stride: int = 20) -> list[int]:
    """Match the current Wenbo script behavior for atlas-derived frames."""
    ply_files = sorted(path for path in atlas_dir.iterdir() if path.suffix == ".ply")
    if not ply_files:
        raise ValueError(f"No .ply files found in atlas directory: {atlas_dir}")

    first_name = ply_files[0].stem
    try:
        start = int(first_name.split("-")[1].lstrip("f"))
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Could not parse first atlas frame id from {first_name}") from exc

    return list(range(start, start + len(ply_files) * stride, stride))

