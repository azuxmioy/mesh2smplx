"""Joint mapping utilities for SMPL/SMPL-X fitting."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class JointMapper(nn.Module):
    def __init__(self, joint_maps: np.ndarray | list[int] | None = None):
        super().__init__()
        if joint_maps is None:
            self.joint_maps = None
        else:
            self.register_buffer("joint_maps", torch.tensor(joint_maps, dtype=torch.long))

    def forward(self, joints: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.joint_maps is None:
            return joints
        return torch.index_select(joints, 1, self.joint_maps)


def smpl_to_openpose(
    model_type: str = "smplx",
    use_hands: bool = True,
    use_face: bool = True,
    use_face_contour: bool = True,
    openpose_format: str = "coco25",
) -> np.ndarray:
    """Return the model-joint indices matching an OpenPose-style target.

    This mirrors the original pipeline's SMPL/SMPL-H/SMPL-X model-to-data
    mappings while keeping them inside the dependency-light package.
    """
    if openpose_format.lower() != "coco25":
        raise ValueError(f"Unsupported OpenPose format: {openpose_format}")

    if model_type == "smpl":
        return np.array(
            [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
            dtype=np.int64,
        )

    if model_type == "smplh":
        body_mapping = np.array(
            [52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62],
            dtype=np.int64,
        )
        mapping = [body_mapping]
        if use_hands:
            mapping.extend(
                [
                    np.array(
                        [20, 34, 35, 36, 63, 22, 23, 24, 64, 25, 26, 27, 65, 31, 32, 33, 66, 28, 29, 30, 67],
                        dtype=np.int64,
                    ),
                    np.array(
                        [21, 49, 50, 51, 68, 37, 38, 39, 69, 40, 41, 42, 70, 46, 47, 48, 71, 43, 44, 45, 72],
                        dtype=np.int64,
                    ),
                ]
            )
        return np.concatenate(mapping)

    if model_type == "smplx":
        body_mapping = np.array(
            [55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65],
            dtype=np.int64,
        )
        mapping = [body_mapping]
        if use_hands:
            mapping.extend(
                [
                    np.array(
                        [20, 37, 38, 39, 66, 25, 26, 27, 67, 28, 29, 30, 68, 34, 35, 36, 69, 31, 32, 33, 70],
                        dtype=np.int64,
                    ),
                    np.array(
                        [21, 52, 53, 54, 71, 40, 41, 42, 72, 43, 44, 45, 73, 49, 50, 51, 74, 46, 47, 48, 75],
                        dtype=np.int64,
                    ),
                ]
            )
        if use_face:
            end_idx = 127 + 17 * int(use_face_contour)
            mapping.append(np.arange(76, end_idx, dtype=np.int64))
        return np.concatenate(mapping)

    raise ValueError(f"Unsupported model type: {model_type}")


def smplx_to_openpose135() -> np.ndarray:
    return smpl_to_openpose("smplx", use_hands=True, use_face=True, use_face_contour=True)
