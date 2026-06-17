"""Parse CMU OpenPose JSON into the 135-keypoint array the fitter consumes.

The layout produced here matches ``fitting.joints.smplx_to_openpose135`` and the
original ``phd.utils.image.load_openpose_json`` contract:

    body (25) + left hand (21) + right hand (21) + face inner (51) + contour (17)

i.e. 135 ``(x, y, confidence)`` rows. This module is dependency-light (numpy +
json only) so it stays inside the core package; the actual detector lives in the
optional vendored :mod:`smpl_registration.keypoints.openpose135` subpackage.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

N_BODY = 25
N_HAND = 21
N_FACE_INNER = 51  # face landmarks 17..67
N_FACE_CONTOUR = 17  # face landmarks 0..16
KEYPOINT_COUNT = N_BODY + 2 * N_HAND + N_FACE_INNER + N_FACE_CONTOUR  # 135


def keypoint_json_path(directory: Path, camera_id: str, frame_id: int) -> Path:
    """Canonical per-camera/per-frame OpenPose JSON path."""
    return Path(directory) / camera_id / f"{frame_id:06d}_keypoints.json"


def load_openpose_json(json_path: str | Path, thres: float = 0.05) -> np.ndarray:
    """Load one person's OpenPose-135 keypoints, zeroing low-confidence rows.

    Returns an array with shape ``(135, 3)``. A missing file or an empty
    ``people`` list yields all-zero keypoints (confidence 0), which the
    triangulator treats as unobserved.
    """
    path = Path(json_path)
    if not path.exists():
        return np.zeros((KEYPOINT_COUNT, 3), dtype=np.float64)

    people = json.loads(path.read_text()).get("people", [])
    if not people:
        return np.zeros((KEYPOINT_COUNT, 3), dtype=np.float64)

    person = people[0]
    body = np.asarray(person["pose_keypoints_2d"], dtype=np.float64).reshape(-1, 3)
    left_hand = np.asarray(person["hand_left_keypoints_2d"], dtype=np.float64).reshape(-1, 3)
    right_hand = np.asarray(person["hand_right_keypoints_2d"], dtype=np.float64).reshape(-1, 3)
    face = np.asarray(person["face_keypoints_2d"], dtype=np.float64).reshape(-1, 3)
    face_inner = face[N_FACE_CONTOUR : N_FACE_CONTOUR + N_FACE_INNER]
    contour = face[:N_FACE_CONTOUR]

    result = np.concatenate([body, left_hand, right_hand, face_inner, contour], axis=0)
    result[result[:, 2] < thres, 2] = 0.0
    return result


def load_frame_keypoints(
    keypoints_dir: str | Path,
    camera_ids: list[str],
    frame_id: int,
    thres: float = 0.05,
) -> np.ndarray:
    """Stack per-camera keypoints for one frame.

    Returns an array with shape ``(num_cameras, 135, 3)`` ordered to match
    ``camera_ids`` so it lines up with projection matrices built in the same
    order for triangulation.
    """
    stacked = [
        load_openpose_json(keypoint_json_path(keypoints_dir, camera_id, frame_id), thres=thres)
        for camera_id in camera_ids
    ]
    return np.stack(stacked, axis=0)
