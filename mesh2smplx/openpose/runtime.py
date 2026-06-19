"""High-level runners for the OpenPose-135 detector.

The OpenPose CLI and any UI wrappers should call into these
helpers rather than open-coding the per-frame loop. Inputs are decoded with
PIL / cv2 and outputs are written as we go (constant memory, video-safe).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

import cv2
import numpy as np
from PIL import Image

from .detector import OpenPose135Detector
from .draw import draw_people

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageTransform:
    crop_box: tuple[int, int, int, int]
    input_size: tuple[int, int]


# ---------- Output sinks ---------------------------------------------------

def _write_json(path: Path, people: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 1.3, "people": people}))


def _write_overlay_png(path: Path, img_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # cv2 expects BGR; img is RGB.
    cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


def preprocess_image(
    img_rgb: np.ndarray,
    *,
    mask_path: Path | None = None,
    crop_to_mask: bool = False,
    crop_padding: float = 0.15,
    crop_padding_pixels: int | None = 30,
    crop_aspect_height: int | None = 1200,
    crop_aspect_width: int | None = 900,
    max_input_size: int | None = None,
) -> tuple[np.ndarray, ImageTransform]:
    """Crop/resize an image for OpenPose and remember how to map points back."""
    height, width = img_rgb.shape[:2]
    crop_box = (0, 0, width, height)
    if crop_to_mask and mask_path is not None and mask_path.exists():
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        if mask.any():
            crop_box = _mask_crop_box(
                mask,
                padding_fraction=crop_padding,
                padding_pixels=crop_padding_pixels,
                aspect_height=crop_aspect_height,
                aspect_width=crop_aspect_width,
            )

    x0, y0, x1, y1 = crop_box
    cropped = img_rgb[y0:y1, x0:x1]
    input_img = _resize_max_side(cropped, max_input_size)
    return input_img, ImageTransform(crop_box=crop_box, input_size=input_img.shape[:2])


def transform_people_to_original(people: list[dict], transform: ImageTransform) -> list[dict]:
    """Map CMU OpenPose 2D keypoints from transformed image pixels to original pixels."""
    x0, y0, x1, y1 = transform.crop_box
    crop_w = max(1, x1 - x0)
    crop_h = max(1, y1 - y0)
    input_h, input_w = transform.input_size
    scale_x = crop_w / max(1, input_w)
    scale_y = crop_h / max(1, input_h)
    fields = (
        "pose_keypoints_2d",
        "hand_left_keypoints_2d",
        "hand_right_keypoints_2d",
        "face_keypoints_2d",
    )

    transformed = []
    for person in people:
        person_out = dict(person)
        for field in fields:
            values = person_out.get(field)
            if not values:
                continue
            keypoints = np.asarray(values, dtype=np.float32).reshape(-1, 3)
            valid = keypoints[:, 2] > 0
            keypoints[valid, 0] = keypoints[valid, 0] * scale_x + x0
            keypoints[valid, 1] = keypoints[valid, 1] * scale_y + y0
            person_out[field] = keypoints.reshape(-1).tolist()
        transformed.append(person_out)
    return transformed


def _mask_crop_box(
    mask: np.ndarray,
    *,
    padding_fraction: float,
    padding_pixels: int | None,
    aspect_height: int | None,
    aspect_width: int | None,
) -> tuple[int, int, int, int]:
    height, width = mask.shape[:2]
    ys, xs = np.nonzero(mask)
    mask_x0 = int(xs.min())
    mask_x1 = int(xs.max()) + 1
    mask_y0 = int(ys.min())
    mask_y1 = int(ys.max()) + 1
    bbox_w = max(1, mask_x1 - mask_x0)
    bbox_h = max(1, mask_y1 - mask_y0)

    if padding_pixels is None:
        pad = int(round(max(bbox_w, bbox_h) * max(0.0, padding_fraction)))
    else:
        pad = max(0, int(padding_pixels))

    wanted_w = bbox_w + 2 * pad
    wanted_h = bbox_h + 2 * pad
    if (
        aspect_height is not None
        and aspect_width is not None
        and aspect_height > 0
        and aspect_width > 0
    ):
        aspect = float(aspect_height) / float(aspect_width)
        wanted_h = max(wanted_h, int(round(wanted_w * aspect)))
        wanted_w = max(wanted_w, int(round(wanted_h / aspect)))

    crop_h = min(height, max(1, wanted_h))
    crop_w = min(width, max(1, wanted_w))
    center_x = (mask_x0 + mask_x1 - 1) / 2.0
    center_y = (mask_y0 + mask_y1 - 1) / 2.0

    x0 = _clamped_start(center_x, crop_w, width)
    y0 = _clamped_start(center_y, crop_h, height)
    return (x0, y0, x0 + crop_w, y0 + crop_h)


def _clamped_start(center: float, length: int, limit: int) -> int:
    start = int(round(center - length / 2.0))
    return min(max(0, start), max(0, limit - length))


def _resize_max_side(img_rgb: np.ndarray, max_input_size: int | None) -> np.ndarray:
    if max_input_size is None or max_input_size <= 0:
        return img_rgb
    height, width = img_rgb.shape[:2]
    max_side = max(height, width)
    if max_side <= max_input_size:
        return img_rgb
    scale = max_input_size / float(max_side)
    out_w = max(1, int(round(width * scale)))
    out_h = max(1, int(round(height * scale)))
    return cv2.resize(img_rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)


# ---------- Image input ----------------------------------------------------

def process_image(
    detector: OpenPose135Detector,
    img_path: Path,
    *,
    write_json: Path | None = None,
    write_image: Path | None = None,
    render_pose: int = 2,
    number_people_max: int = 0,
    mask_path: Path | None = None,
    crop_to_mask: bool = False,
    crop_padding: float = 0.15,
    crop_padding_pixels: int | None = 30,
    crop_aspect_height: int | None = 1200,
    crop_aspect_width: int | None = 900,
    max_input_size: int | None = None,
) -> tuple[list[dict], np.ndarray]:
    """Run on one image, optionally writing JSON + overlay PNG."""
    img = np.asarray(Image.open(img_path).convert("RGB"))
    detector_input, transform = preprocess_image(
        img,
        mask_path=mask_path,
        crop_to_mask=crop_to_mask,
        crop_padding=crop_padding,
        crop_padding_pixels=crop_padding_pixels,
        crop_aspect_height=crop_aspect_height,
        crop_aspect_width=crop_aspect_width,
        max_input_size=max_input_size,
    )
    people = detector(detector_input, number_people_max=number_people_max)
    people = transform_people_to_original(people, transform)
    if write_json is not None:
        _write_json(write_json, people)
    overlay = draw_people(img, people, render_pose=render_pose) if render_pose > 0 else img
    if write_image is not None:
        _write_overlay_png(write_image, overlay)
    return people, overlay


def process_image_dir(
    detector: OpenPose135Detector,
    image_dir: Path,
    *,
    write_json_dir: Path | None = None,
    write_image_dir: Path | None = None,
    render_pose: int = 2,
    number_people_max: int = 0,
    overwrite: bool = False,
    max_input_size: int | None = None,
    progress: Callable[[Iterable], Iterable] | None = None,
) -> None:
    """Walk image_dir, run on each frame, write outputs alongside as named `<stem>...`."""
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise ValueError(f"No images found in {image_dir}")

    iterator: Iterable = images
    if progress is not None:
        iterator = progress(images)

    for img_path in iterator:
        json_path = (write_json_dir / f"{img_path.stem}_keypoints.json") if write_json_dir else None
        img_out_path = (write_image_dir / f"{img_path.stem}.png") if write_image_dir else None
        if (
            not overwrite
            and json_path
            and json_path.exists()
            and (img_out_path is None or img_out_path.exists())
        ):
            continue
        process_image(
            detector, img_path,
            write_json=json_path,
            write_image=img_out_path,
            render_pose=render_pose,
            number_people_max=number_people_max,
            max_input_size=max_input_size,
        )


# ---------- Video input ----------------------------------------------------

def _video_frames(cap: cv2.VideoCapture) -> Iterator[tuple[int, np.ndarray]]:
    idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            return
        yield idx, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        idx += 1


def process_video(
    detector: OpenPose135Detector,
    video_path: Path,
    *,
    write_json_dir: Path | None = None,
    write_image_dir: Path | None = None,
    write_video: Path | None = None,
    render_pose: int = 2,
    number_people_max: int = 0,
    progress: Callable[[Iterable], Iterable] | None = None,
) -> None:
    """Stream frames out of `video_path`, run on each, write outputs as we go."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if write_video is not None:
        write_video.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(write_video), fourcc, fps, (W, H))

    try:
        iterator: Iterable = _video_frames(cap)
        if progress is not None:
            iterator = progress(iterator)

        for idx, frame_rgb in iterator:
            people = detector(frame_rgb, number_people_max=number_people_max)
            if write_json_dir is not None:
                _write_json(write_json_dir / f"frame_{idx:06d}_keypoints.json", people)
            overlay = (
                draw_people(frame_rgb, people, render_pose=render_pose)
                if render_pose > 0
                else frame_rgb
            )
            if write_image_dir is not None:
                _write_overlay_png(write_image_dir / f"frame_{idx:06d}.png", overlay)
            if writer is not None:
                writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    finally:
        cap.release()
        if writer is not None:
            writer.release()
