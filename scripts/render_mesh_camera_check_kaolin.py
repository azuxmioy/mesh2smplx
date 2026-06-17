"""Render a textured mesh through calibrated OpenCV cameras with Kaolin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--camera-json", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--frame-id", type=int, required=True)
    parser.add_argument("--cameras", default="0,24,40,61")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render-scale", type=float, default=0.25)
    parser.add_argument("--overlay-alpha", type=float, default=0.55)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", default="cuda", choices=("cuda", "nvdiffrast", "nvdiffrast_fwd"))
    parser.add_argument(
        "--y-axis",
        default="up",
        choices=("down", "up"),
        help="Kaolin NDC y convention. Use 'down' if the render appears vertically flipped.",
    )
    return parser.parse_args()


def parse_camera_ids(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def require_kaolin():
    try:
        import kaolin as kal
    except ImportError as exc:
        raise SystemExit(
            "This render check requires Kaolin. Install the project render extra on a "
            "GPU environment, then rerun this script."
        ) from exc
    return kal


def main() -> None:
    import torch
    import trimesh

    args = parse_args()
    kal = require_kaolin()
    device = torch.device(args.device)
    mesh = trimesh.load(args.mesh, force="mesh", process=False)
    if mesh.is_empty:
        raise ValueError(f"Could not load mesh from {args.mesh}")
    if getattr(mesh.visual, "uv", None) is None:
        raise ValueError(f"Mesh does not contain UV coordinates: {args.mesh}")

    texture_image = getattr(getattr(mesh.visual, "material", None), "image", None)
    if texture_image is None:
        raise ValueError(f"Mesh material does not contain a texture image: {args.mesh}")

    vertices = torch.as_tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=device)
    faces = torch.as_tensor(np.asarray(mesh.faces), dtype=torch.long, device=device)
    uv = torch.as_tensor(np.asarray(mesh.visual.uv), dtype=torch.float32, device=device)
    texture = image_to_texture_tensor(texture_image).to(device)

    cameras = json.loads(args.camera_json.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for camera_id in parse_camera_ids(args.cameras):
        if camera_id not in cameras:
            print(f"skip missing camera {camera_id}")
            continue
        image_path = args.image_root / camera_id / f"{args.frame_id:06d}.png"
        if not image_path.exists():
            print(f"skip missing image {image_path}")
            continue

        image = Image.open(image_path).convert("RGB")
        rendered, mask = render_camera_view(
            kal=kal,
            vertices=vertices,
            faces=faces,
            uv=uv,
            texture=texture,
            camera=cameras[camera_id],
            image_size=image.size,
            render_scale=args.render_scale,
            backend=args.backend,
            y_axis=args.y_axis,
        )
        image_small = image.resize((rendered.shape[1], rendered.shape[0]), Image.Resampling.BILINEAR)
        overlay = blend_overlay(np.asarray(image_small), rendered, mask, args.overlay_alpha)
        comparison = np.concatenate([np.asarray(image_small), rendered, overlay], axis=1)

        stem = f"frame_{args.frame_id:06d}_cam_{int(camera_id):03d}"
        render_path = args.output_dir / f"{stem}_render.png"
        overlay_path = args.output_dir / f"{stem}_overlay.png"
        comparison_path = args.output_dir / f"{stem}_comparison.png"
        Image.fromarray(rendered).save(render_path)
        Image.fromarray(overlay).save(overlay_path)
        Image.fromarray(comparison).save(comparison_path)

        ys, xs = np.where(mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None
        stats = {
            "camera": camera_id,
            "coverage": float(mask.mean()),
            "bbox": bbox,
            "image": str(image_path),
            "render": str(render_path),
            "overlay": str(overlay_path),
            "comparison": str(comparison_path),
        }
        summary.append(stats)
        print(f"camera {camera_id}: coverage={stats['coverage']:.4f} bbox={bbox}")

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def image_to_texture_tensor(image: Image.Image):
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
    camera: dict,
    image_size: tuple[int, int],
    render_scale: float,
    backend: str,
    y_axis: str,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    image_width, image_height = image_size
    calib_rows, calib_cols = camera["shape"]
    scale_x = image_width / float(calib_cols)
    scale_y = image_height / float(calib_rows)
    width = _multiple_of_8(max(8, int(round(image_width * render_scale))))
    height = _multiple_of_8(max(8, int(round(image_height * render_scale))))

    intrinsics = np.asarray(camera["intrinsics"], dtype=np.float32).copy()
    intrinsics[0, :] *= scale_x * (width / image_width)
    intrinsics[1, :] *= scale_y * (height / image_height)
    extrinsics = np.asarray(camera["extrinsics"], dtype=np.float32)

    intrinsics_t = torch.as_tensor(intrinsics, dtype=torch.float32, device=vertices.device)
    extrinsics_t = torch.as_tensor(extrinsics, dtype=torch.float32, device=vertices.device)
    camera_vertices = vertices @ extrinsics_t[:, :3].T + extrinsics_t[:, 3]
    z = camera_vertices[:, 2].clamp_min(1e-6)

    x_pixels = (intrinsics_t[0, 0] * camera_vertices[:, 0] + intrinsics_t[0, 2] * z) / z
    y_pixels = (intrinsics_t[1, 1] * camera_vertices[:, 1] + intrinsics_t[1, 2] * z) / z
    x_ndc = (2.0 * x_pixels / max(1, width - 1)) - 1.0
    if y_axis == "down":
        y_ndc = (2.0 * y_pixels / max(1, height - 1)) - 1.0
    else:
        y_ndc = 1.0 - (2.0 * y_pixels / max(1, height - 1))

    face_vertices_z = -z[faces].unsqueeze(0)
    face_vertices_image = torch.stack((x_ndc, y_ndc), dim=-1)[faces].unsqueeze(0)
    face_uv = uv[faces].unsqueeze(0)
    valid_faces = torch.all(z[faces] > 1e-6, dim=-1).unsqueeze(0)

    uv_image, face_idx = kal.render.mesh.rasterize(
        height,
        width,
        face_vertices_z,
        face_vertices_image,
        face_uv,
        valid_faces=valid_faces,
        backend=backend,
    )
    rgb = kal.render.mesh.texture_mapping(uv_image, texture, mode="bilinear")
    mask = face_idx[0] >= 0
    rgb = torch.where(mask.unsqueeze(-1), rgb[0], torch.zeros_like(rgb[0]))
    rendered = (rgb.clamp(0.0, 1.0).detach().cpu().numpy() * 255.0).astype(np.uint8)
    return rendered, mask.detach().cpu().numpy()


def _multiple_of_8(value: int) -> int:
    return int(np.ceil(value / 8.0) * 8)


def blend_overlay(image: np.ndarray, rendered: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    overlay = image.copy()
    overlay[mask] = (
        (1.0 - alpha) * overlay[mask].astype(np.float32)
        + alpha * rendered[mask].astype(np.float32)
    ).astype(np.uint8)
    overlay[mask & (rendered.sum(axis=-1) == 0)] = image[mask & (rendered.sum(axis=-1) == 0)]
    return overlay


if __name__ == "__main__":
    main()
