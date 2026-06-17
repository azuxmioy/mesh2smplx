"""Dependency-light triangulation utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .data.interfaces import CameraModel


def projection_matrix(camera: "CameraModel") -> np.ndarray:
    """Return the 3x4 projection matrix ``K @ [R|t]`` for a camera."""
    intrinsics = np.asarray(camera.intrinsics, dtype=np.float64)
    extrinsics = np.asarray(camera.extrinsics, dtype=np.float64)
    return intrinsics @ extrinsics


def triangulate_frame(
    cameras: list["CameraModel"],
    keypoints: np.ndarray,
    reg: float = 1e-3,
):
    """Triangulate keypoints observed by an ordered list of cameras.

    Args:
        cameras: Cameras in the same order as the first axis of ``keypoints``.
        keypoints: Array ``(num_cameras, num_joints, 3)`` of ``(x, y, conf)``.

    Returns:
        Tuple ``(keypoints_3d, average_reprojection_error)`` from
        :func:`triangulate_dlt`.
    """
    projections = np.stack([projection_matrix(camera) for camera in cameras], axis=0)
    return triangulate_dlt(projections, np.asarray(keypoints, dtype=np.float64), reg=reg)


def triangulate_dlt(projection_matrices: np.ndarray, keypoints: np.ndarray, reg: float = 1e-3):
    """Triangulate weighted 2D keypoints with regularized least squares.

    Args:
        projection_matrices: Array with shape `(num_cameras, 3, 4)`.
        keypoints: Array with shape `(num_cameras, num_joints, 3)`, where the
            last coordinate is confidence.
        reg: Diagonal regularization for under-observed joints.

    Returns:
        Tuple of `(keypoints_3d, average_reprojection_error)`.
    """
    joint_count = keypoints.shape[1]
    normal = np.zeros((joint_count, 3, 4), dtype=np.float64)
    normal[:] = np.eye(3, 4) * reg

    for projection, camera_keypoints in zip(projection_matrices, keypoints):
        x_coord, y_coord, weight = camera_keypoints.T
        rows = np.stack(
            [
                x_coord[:, None] * projection[2] - projection[0],
                y_coord[:, None] * projection[2] - projection[1],
            ],
            axis=1,
        )
        normal += rows.transpose(0, 2, 1)[:, :-1, :] @ rows * weight[:, None, None]

    confidence = keypoints[:, :, 2]
    mean_confidence = np.sum(confidence, axis=0) / np.maximum(np.sum(confidence > 0, axis=0), 1)

    keypoints_3d = np.ones((joint_count, 5), dtype=np.float64)
    keypoints_3d[:, :3] = np.linalg.solve(normal[:, :, :-1], -normal[:, :, -1])
    keypoints_3d[:, 4] = mean_confidence

    reprojection_errors = []
    reprojection_weights = []
    for projection, camera_keypoints in zip(projection_matrices, keypoints):
        projected = np.einsum("ij,kj->ik", keypoints_3d[:, :4], projection)
        projected /= projected[:, 2, None]
        error = np.sqrt(((projected[:, :2] - camera_keypoints[:, :2]) ** 2).sum(axis=-1))
        reprojection_errors.append(error)
        reprojection_weights.append(camera_keypoints[:, 2])

    errors = np.stack(reprojection_errors, axis=0)
    weights = np.stack(reprojection_weights, axis=0)
    average_error = (errors * weights).sum(axis=0) / np.maximum(weights.sum(axis=0), 1e-8)
    return keypoints_3d, float(average_error.mean())

