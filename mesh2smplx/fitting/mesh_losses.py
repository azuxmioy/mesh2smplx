"""Dependency-light mesh losses used by the full fitting schedule."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class MeshTarget:
    points: torch.Tensor
    vertices: torch.Tensor
    faces: torch.Tensor


def load_mesh_target(
    mesh_path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    scale: float = 1.0,
    samples: int = 5000,
    seed: int = 0,
) -> MeshTarget:
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("Mesh fitting requires `trimesh`.") from exc

    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if mesh.is_empty:
        raise ValueError(f"Could not load mesh geometry from {mesh_path}")

    vertices = torch.as_tensor(mesh.vertices * scale, dtype=dtype, device=device)
    faces = torch.as_tensor(mesh.faces, dtype=torch.long, device=device)

    sampled, _ = trimesh.sample.sample_surface(mesh, samples, seed=seed)
    sampled = sampled * scale
    points = torch.as_tensor(sampled, dtype=dtype, device=device).unsqueeze(0)
    return MeshTarget(points=points, vertices=vertices.unsqueeze(0), faces=faces)


def symmetric_chamfer_loss(
    vertices: torch.Tensor,
    target_points: torch.Tensor,
    *,
    vertex_samples: int = 5000,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Approximate symmetric scan/body distance.

    `vertices` remains differentiable. `target_points` is treated as a fixed
    sampled scan point cloud.
    """
    sampled_vertices = _subsample_vertices(vertices, vertex_samples)
    source_to_target = _mean_min_squared_distance(sampled_vertices, target_points, chunk_size)
    target_to_source = _mean_min_squared_distance(target_points, sampled_vertices, chunk_size)
    return source_to_target + target_to_source


def _subsample_vertices(vertices: torch.Tensor, count: int) -> torch.Tensor:
    if vertices.shape[1] <= count:
        return vertices
    indices = torch.linspace(
        0,
        vertices.shape[1] - 1,
        count,
        device=vertices.device,
    ).long()
    return vertices[:, indices]


def _mean_min_squared_distance(source: torch.Tensor, target: torch.Tensor, chunk_size: int) -> torch.Tensor:
    mins = []
    for start in range(0, source.shape[1], chunk_size):
        chunk = source[:, start : start + chunk_size]
        distances = torch.cdist(chunk, target).square()
        mins.append(distances.min(dim=-1).values)
    return torch.cat(mins, dim=1).mean()
