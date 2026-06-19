"""Virtual-camera observation records for textured mesh sequences.

Kaolin is intentionally optional and imported only inside rendering helpers.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import InputConfig, VirtualCameraConfig
from ..data.camera_io import load_camera_models, resolve_camera_json
from ..data.interfaces import CameraModel, FrameObservation, ObservationBundle
from ..data.mesh_sequence import MeshFrame

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def require_kaolin():
    try:
        import kaolin
    except ImportError as exc:
        raise RuntimeError(
            "Virtual-camera rendering requires the optional render extra. "
            "Install Kaolin for the PyTorch/CUDA version on your server."
        ) from exc
    return kaolin


def build_orbit_camera_ids(count: int) -> list[str]:
    return [f"virtual_{idx:03d}" for idx in range(count)]


def placeholder_virtual_cameras(
    config: VirtualCameraConfig,
    center: np.ndarray | None = None,
    radius: float | None = None,
) -> dict[str, CameraModel]:
    """Create upper-semi-sphere virtual camera records."""
    cameras: dict[str, CameraModel] = {}
    center = np.zeros(3, dtype=np.float32) if center is None else center.astype(np.float32)
    radius = float(radius if radius is not None else 2.5)
    elevation_min = np.deg2rad(max(0.0, config.elevation_degrees))
    elevation_max = np.deg2rad(75.0)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for index, camera_id in enumerate(build_orbit_camera_ids(config.count)):
        t = (index + 0.5) / max(1, config.count)
        elevation = elevation_min + (elevation_max - elevation_min) * t
        azimuth = golden_angle * index + np.deg2rad(config.azimuth_offset_degrees)
        position = center + radius * np.array(
            [
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
            ],
            dtype=np.float32,
        )
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
            extrinsics=_look_at_extrinsics(position, center),
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

    observations = []
    image_root = camera_config.output_dir / "images"
    mask_root = camera_config.output_dir / "masks"
    using_calibrated_cameras = input_config.cameras is not None
    if using_calibrated_cameras:
        camera_json = resolve_camera_json(input_config.cameras)
        cameras = load_camera_models(camera_json, images_root=None)
        print(f"render_cameras=calibration path={camera_json} count={len(cameras)}")
    else:
        center, radius = _estimate_orbit_from_meshes(input_config, camera_config, mesh_frames)
        cameras = placeholder_virtual_cameras(camera_config, center=center, radius=radius)
        print(f"render_cameras=heuristic count={len(cameras)}")
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
    bundle = ObservationBundle(cameras=cameras, frames=observations)
    _ensure_rendered_images(
        bundle=bundle,
        mesh_frames=mesh_frames,
        render_masks=camera_config.render_masks,
        background_color=camera_config.background_color,
        scale=1.0 if using_calibrated_cameras else input_config.scale_to_meters,
    )
    return bundle


def _ensure_rendered_images(
    *,
    bundle: ObservationBundle,
    mesh_frames: list[MeshFrame],
    render_masks: bool,
    background_color: tuple[float, float, float],
    scale: float,
) -> None:
    if not _has_missing_render_outputs(bundle, render_masks):
        return

    import torch

    kal = require_kaolin()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Virtual mesh rendering requires CUDA because it uses Kaolin rasterization."
        )
    device = torch.device("cuda")
    frame_by_id = {frame.frame_id: frame for frame in bundle.frames}
    rendered_count = 0

    for mesh_frame in mesh_frames:
        frame = frame_by_id[mesh_frame.frame_id]
        if not _frame_needs_render(frame, render_masks):
            continue
        vertices, faces, uv, texture = _load_render_mesh(mesh_frame, scale=scale, device=device)
        for camera_id, camera in bundle.cameras.items():
            image_path = frame.image_paths[camera_id]
            mask_path = frame.mask_paths.get(camera_id) if frame.mask_paths is not None else None
            if image_path.exists() and (
                not render_masks or mask_path is None or mask_path.exists()
            ):
                continue

            rendered, mask = render_camera_view(
                kal=kal,
                vertices=vertices,
                faces=faces,
                uv=uv,
                texture=texture,
                camera=camera,
                background_color=background_color,
            )
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rendered).save(image_path)
            if render_masks and mask_path is not None:
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(mask_path)
            rendered_count += 1

    if rendered_count:
        print(f"rendered_virtual_images={rendered_count}")


def _has_missing_render_outputs(bundle: ObservationBundle, render_masks: bool) -> bool:
    return any(_frame_needs_render(frame, render_masks) for frame in bundle.frames)


def _frame_needs_render(frame: FrameObservation, render_masks: bool) -> bool:
    for camera_id, image_path in frame.image_paths.items():
        if not image_path.exists():
            return True
        if render_masks and frame.mask_paths is not None:
            mask_path = frame.mask_paths.get(camera_id)
            if mask_path is not None and not mask_path.exists():
                return True
    return False


def _load_render_mesh(mesh_frame: MeshFrame, *, scale: float, device):
    import torch
    import trimesh

    mesh = trimesh.load(mesh_frame.mesh_path, force="mesh", process=False)
    if mesh.is_empty:
        raise ValueError(f"Could not load mesh from {mesh_frame.mesh_path}")

    vertices = torch.as_tensor(
        np.asarray(mesh.vertices, dtype=np.float32) * float(scale),
        dtype=torch.float32,
        device=device,
    )
    faces = torch.as_tensor(np.asarray(mesh.faces), dtype=torch.long, device=device)

    uv_array = getattr(mesh.visual, "uv", None)
    uv = None
    if uv_array is not None:
        uv = torch.as_tensor(np.asarray(uv_array), dtype=torch.float32, device=device)

    texture_image = None
    if mesh_frame.texture_path is not None:
        texture_image = Image.open(mesh_frame.texture_path)
    else:
        texture_path = _find_texture_for_mesh(mesh_frame)
        if texture_path is not None:
            texture_image = Image.open(texture_path)
    if texture_image is None:
        texture_image = getattr(getattr(mesh.visual, "material", None), "image", None)
    texture = (
        _image_to_texture_tensor(texture_image).to(device)
        if texture_image is not None
        else None
    )
    return vertices, faces, uv, texture


def _find_texture_for_mesh(mesh_frame: MeshFrame) -> Path | None:
    texture_path = _find_texture_from_obj_material(mesh_frame.mesh_path)
    if texture_path is not None:
        return texture_path

    mesh_frame_id = mesh_frame.frame_id
    candidates = [
        path
        for path in mesh_frame.mesh_path.parent.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    for path in sorted(candidates):
        if _parse_last_int(path.stem) == mesh_frame_id:
            return path
    return None


def _find_texture_from_obj_material(mesh_path: Path) -> Path | None:
    if mesh_path.suffix.lower() != ".obj":
        return None
    try:
        lines = mesh_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None

    material_files = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("mtllib "):
            material_files.extend(stripped.split()[1:])
    if not material_files:
        material_files = [path.name for path in mesh_path.parent.glob("*.mtl")]

    for material_file in material_files:
        material_path = _case_insensitive_child(mesh_path.parent, material_file)
        if material_path is None or not material_path.exists():
            continue
        texture_path = _find_texture_from_mtl(material_path)
        if texture_path is not None:
            return texture_path
    return None


def _find_texture_from_mtl(material_path: Path) -> Path | None:
    try:
        lines = material_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if tokens[0].lower() != "map_kd" or len(tokens) < 2:
            continue
        texture_name = tokens[-1]
        texture_path = _case_insensitive_child(material_path.parent, texture_name)
        if texture_path is not None and texture_path.exists():
            return texture_path
    return None


def _case_insensitive_child(parent: Path, name: str) -> Path | None:
    candidate = parent / name
    if candidate.exists():
        return candidate
    lookup = name.lower()
    try:
        for child in parent.iterdir():
            if child.name.lower() == lookup:
                return child
    except OSError:
        return None
    return None


def _parse_last_int(text: str) -> int | None:
    matches = re.findall(r"(\d+)", text)
    return int(matches[-1]) if matches else None


def _image_to_texture_tensor(image: Image.Image):
    import torch

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()


def render_camera_view(
    *,
    kal,
    vertices,
    faces,
    uv,
    texture,
    camera: CameraModel,
    background_color: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    image_height, image_width = camera.image_size
    intrinsics_t = torch.as_tensor(camera.intrinsics, dtype=torch.float32, device=vertices.device)
    extrinsics_t = torch.as_tensor(camera.extrinsics, dtype=torch.float32, device=vertices.device)
    camera_vertices = vertices @ extrinsics_t[:, :3].T + extrinsics_t[:, 3]
    z = camera_vertices[:, 2]
    z_safe = z.clamp_min(1e-6)

    x_pixels = (
        intrinsics_t[0, 0] * camera_vertices[:, 0] + intrinsics_t[0, 2] * z_safe
    ) / z_safe
    y_pixels = (
        intrinsics_t[1, 1] * camera_vertices[:, 1] + intrinsics_t[1, 2] * z_safe
    ) / z_safe
    x_ndc = (2.0 * x_pixels / max(1, image_width - 1)) - 1.0
    y_ndc = 1.0 - (2.0 * y_pixels / max(1, image_height - 1))

    face_vertices_z = -z_safe[faces].unsqueeze(0)
    face_vertices_image = torch.stack((x_ndc, y_ndc), dim=-1)[faces].unsqueeze(0)
    valid_faces = torch.all(z[faces] > 1e-6, dim=-1).unsqueeze(0)

    if uv is not None and texture is not None:
        face_features = uv[faces].unsqueeze(0)
        uv_image, face_idx = kal.render.mesh.rasterize(
            image_height,
            image_width,
            face_vertices_z,
            face_vertices_image,
            face_features,
            valid_faces=valid_faces,
            backend="cuda",
        )
        rgb = kal.render.mesh.texture_mapping(uv_image, texture, mode="bilinear")[0]
    else:
        base_color = torch.tensor([0.74, 0.70, 0.64], dtype=torch.float32, device=vertices.device)
        face_features = base_color.view(1, 1, 1, 3).expand(1, faces.shape[0], 3, 3)
        rgb, face_idx = kal.render.mesh.rasterize(
            image_height,
            image_width,
            face_vertices_z,
            face_vertices_image,
            face_features,
            valid_faces=valid_faces,
            backend="cuda",
        )
        rgb = rgb[0]

    mask = face_idx[0] >= 0
    background = torch.as_tensor(background_color, dtype=torch.float32, device=vertices.device)
    background = background.view(1, 1, 3).expand_as(rgb)
    rgb = torch.where(mask.unsqueeze(-1), rgb, background)
    rendered = (rgb.clamp(0.0, 1.0).detach().cpu().numpy() * 255.0).astype(np.uint8)
    return rendered, mask.detach().cpu().numpy()


def _look_at_extrinsics(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - position
    forward = forward / max(float(np.linalg.norm(forward)), 1e-8)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(forward, world_up))) > 0.95:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(world_up, forward)
    right = right / max(float(np.linalg.norm(right)), 1e-8)
    up = np.cross(forward, right)
    rotation = np.stack([right, up, forward], axis=0)
    translation = -rotation @ position
    return np.concatenate([rotation, translation[:, None]], axis=1).astype(np.float32)


def _estimate_orbit_from_meshes(
    input_config: InputConfig,
    camera_config: VirtualCameraConfig,
    mesh_frames: list[MeshFrame],
) -> tuple[np.ndarray, float]:
    center = np.zeros(3, dtype=np.float32)
    radius = 2.5
    if mesh_frames:
        try:
            import trimesh

            mesh = trimesh.load(mesh_frames[0].mesh_path, process=False)
            bounds = np.asarray(mesh.bounds, dtype=np.float32) * input_config.scale_to_meters
            center = bounds.mean(axis=0)
            extent = float(np.linalg.norm(bounds[1] - bounds[0]))
            radius = max(1.0, extent * 1.5)
        except Exception as exc:
            print(f"Warning: could not estimate mesh bounds for virtual cameras: {exc}")
    if not isinstance(camera_config.radius, str):
        radius = float(camera_config.radius)
    return center, radius
