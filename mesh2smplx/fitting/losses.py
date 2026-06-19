"""Small PyTorch losses for the minimal fitting core.

The keypoint objective, body/hand pose priors, joint weights, and loss-weight
schedule below are faithful ports of the original fitting code so this package
keeps the same optimization behavior: anti-hyperextension priors, progressive
joint exposure weighting, and confidence-weighted keypoint terms.
"""

from __future__ import annotations

import torch

# Original per-joint weights (lib/body_objectives.py). First 25 entries follow
# the BODY_25 pattern; every joint beyond that (hands/face/contour) uses 0.6.
_BODY25_WEIGHTS = (
    0.6, 0.6,
    1.0, 1.0, 1.0,  # right arm
    1.0, 1.0, 1.0,  # left arm
    0.6, 0.6,
    1.0, 1.0,       # right knee and leg
    1.0, 1.0,       # left knee and leg
    0.8,            # left foot
    1.0, 1.0, 1.0, 1.0,  # head
    0.8, 0.8, 0.8,  # left foot
    0.8, 0.8, 0.8,  # right foot
)


def original_joint_weights(num_joints: int, device: torch.device | None = None) -> torch.Tensor:
    """Return the legacy (num_joints, 1) joint-weight vector."""
    weights = torch.full((num_joints, 1), 0.6, dtype=torch.float32, device=device)
    upper = min(num_joints, len(_BODY25_WEIGHTS))
    weights[:upper, 0] = torch.tensor(_BODY25_WEIGHTS[:upper], dtype=torch.float32, device=device)
    return weights


def keypoint_objective(
    observed: torch.Tensor,
    predicted: torch.Tensor,
    joint_weights: torch.Tensor,
) -> torch.Tensor:
    """Confidence- and joint-weighted 3D keypoint loss (batch_smpl_3djoints_loss).

    ``observed`` is ``(B, N, >=4)`` with confidence in the last channel (col 4
    for the (x, y, z, 1, conf) triangulation layout, else col 3); ``predicted``
    is ``(B, N, 3)``; ``joint_weights`` is ``(N, 1)``.
    """
    confidence = observed[..., 4] if observed.shape[-1] > 4 else observed[..., 3]
    squared = ((observed[..., :3] - predicted) ** 2).sum(dim=-1)  # (B, N)
    weighted = torch.matmul(squared * confidence, joint_weights)  # (B, 1)
    return weighted.mean()


def body_pose_prior(body_pose: torch.Tensor) -> torch.Tensor:
    """Legacy SMPL/-H/-X body-pose prior with per-joint exp barriers.

    ``body_pose`` is axis-angle ``(B, 63)`` (21 body joints). Verbatim port of
    ``BaseFitter.body_pose_prior``: strong penalties on ankle/elbow/foot, plus
    soft exp barriers keeping wrists, knees and elbows in plausible ranges.
    """
    return torch.sum(body_pose ** 2) + (
        torch.sum(body_pose[:, [20, 23]] ** 2)  # left/right ankle-ish
        + torch.sum(body_pose[:, 27:33] ** 2)   # feet
    ) * 10 ** 5 + (
        torch.sum(body_pose[:, [1, 10, 13, 36, 39, 45, 48, 51, 54]] ** 2)  # collars/shoulders/elbows
        + torch.sum(torch.exp(-3.14 / 2 - body_pose[:, [57, 60]]) + torch.exp(-3.14 / 3 + body_pose[:, [57, 60]]))  # wrists
        + torch.sum(torch.exp(-body_pose[:, [9, 12]]))  # knees
        + torch.sum(torch.exp(body_pose[:, [52]]))      # left elbow
        + torch.sum(torch.exp(-body_pose[:, [55]]))     # right elbow
    )


def _body_pose_joint_l2(body_pose: torch.Tensor, joint_ids: tuple[int, ...]) -> torch.Tensor:
    pose = body_pose.reshape(body_pose.shape[0], -1, 3)
    available_ids = [joint_id for joint_id in joint_ids if joint_id < pose.shape[1]]
    if not available_ids:
        return body_pose.new_zeros(())
    return torch.sum(pose[:, available_ids] ** 2)


def spine_pose_prior(body_pose: torch.Tensor) -> torch.Tensor:
    """Extra L2 regularizer for SMPL spine joints.

    ``body_pose`` excludes the root/global orientation. The zero-based body-pose
    joint ids 2, 5, and 8 correspond to spine1, spine2, and spine3.
    """
    return _body_pose_joint_l2(body_pose, (2, 5, 8))


def neck_pose_prior(body_pose: torch.Tensor) -> torch.Tensor:
    """Extra L2 regularizer for SMPL neck/head joints."""
    return _body_pose_joint_l2(body_pose, (11, 14))


def hand_pose_prior(hand_pose: torch.Tensor) -> torch.Tensor:
    """Legacy per-axis exp-barrier hand-pose prior (BaseFitter.hand_pose_prior)."""
    ratio = 1
    epsilon = 0
    return (
        torch.sum(
            torch.exp(ratio * (-epsilon - 3.14 / 12 - hand_pose[:, [0, 3, 6, 9, 12, 15, 27, 30, 33]]))
            + torch.exp(ratio * (-epsilon - 3.14 / 12 + hand_pose[:, [0, 3, 6, 9, 12, 15, 27, 30, 33]]))
        )
        + torch.sum(
            torch.exp(ratio * (-epsilon - 3.14 / 18 - hand_pose[:, [4, 7, 13, 16, 22, 25, 31, 34]]))
            + torch.exp(ratio * (-epsilon - 3.14 / 18 + hand_pose[:, [4, 7, 13, 16, 22, 25, 31, 34]]))
        )
        + torch.sum(
            torch.exp(ratio * (-epsilon - 3.14 / 9 - hand_pose[:, [1, 10, 19, 28]]))
            + torch.exp(ratio * (-epsilon - 3.14 / 9 + hand_pose[:, [1, 10, 19, 28]]))
        )
        + torch.sum(
            torch.exp(ratio * (-epsilon - 3.14 / 2 - hand_pose[:, list(range(2, 36, 3))]))
            + torch.exp(ratio * (-epsilon - 3.14 / 9 + hand_pose[:, list(range(2, 36, 3))]))
        )
        + torch.sum(
            torch.exp(ratio * (-epsilon - 3.14 / 3 - hand_pose[:, [36, 40]]))
            + torch.exp(ratio * (-epsilon - 3.14 + hand_pose[:, [36, 40]]))
        )
        + torch.sum(
            torch.exp(ratio * (-epsilon - 3.14 / 3 - hand_pose[:, [37, 39, 42]]))
            + torch.exp(ratio * (-epsilon - 3.14 / 3 * 2 + hand_pose[:, [37, 39, 42]]))
        )
        + torch.sum(
            torch.exp(ratio * (-epsilon - 3.14 / 2 - hand_pose[:, [38, 41, 44]]))
            + torch.exp(ratio * (-epsilon - 3.14 / 3 + hand_pose[:, [38, 41, 44]]))
        )
        + torch.sum(
            torch.exp(ratio * (-epsilon - 3.14 / 3 - hand_pose[:, [18, 21, 24]]))
            + torch.exp(ratio * (-epsilon - 3.14 / 9 + hand_pose[:, [18, 21, 24]]))
        )
        + torch.sum(
            torch.exp(ratio * (-epsilon - 3.14 / 9 - hand_pose[:, [43]]))
            + torch.exp(ratio * (-epsilon - 3.14 / 2 + hand_pose[:, [43]]))
        )
    )


def loss_weights(base: dict[str, float] | None = None) -> dict:
    """Build the legacy ``/(1+it)`` weight schedule from base multipliers.

    ``base`` provides the per-term constants (typically ``fitting.loss_weights``
    from the config). The ``jaw`` term keeps the original behaviour of no
    ``/(1+it)`` decay. Missing keys fall back to the legacy defaults.
    """
    b = {
        "limb": 1e4, "betas": 1.0, "pose_pr": 1.0, "lhand": 0.1, "rhand": 0.1,
        "spine_pose": 25.0, "neck_pose": 25.0, "jaw": 1.0, "f_exp": 0.01,
        "pose_obj": 1e5, "icp": 50.0,
    }
    if base:
        b.update(base)
    return {
        "limb_loss": lambda cst, it, w=b["limb"]: w * cst / (1 + it),
        "limb": lambda cst, it, w=b["limb"]: w * cst / (1 + it),
        "reg_loss": lambda x, it: 1e-6 * x,
        "betas": lambda cst, it, w=b["betas"]: w * cst / (1 + it),
        "pose_pr": lambda cst, it, w=b["pose_pr"]: w * cst / (1 + it),
        "spine_pose": lambda cst, it, w=b["spine_pose"]: w * cst / (1 + it),
        "neck_pose": lambda cst, it, w=b["neck_pose"]: w * cst / (1 + it),
        "lhand": lambda cst, it, w=b["lhand"]: w * cst / (1 + it),
        "rhand": lambda cst, it, w=b["rhand"]: w * cst / (1 + it),
        "jaw": lambda cst, it, w=b["jaw"]: w * cst,
        "f_exp": lambda cst, it, w=b["f_exp"]: w * cst / (1 + it),
        "pose_obj": lambda cst, it, w=b["pose_obj"]: w * cst / (1 + it),
        "icp": lambda cst, it, w=b["icp"]: w * cst / (1 + it),
    }
