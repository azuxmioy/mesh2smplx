"""Model-aware fitting schedules.

The original code used different stage sequences for SMPL, SMPL-H, and SMPL-X.
This module makes those schedules explicit so the fitter can preserve the
intended optimization structure without keeping old monolithic classes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FitStage:
    name: str
    iterations: int
    steps_per_iter: int
    optimize: tuple[str, ...]
    joint_slice: slice | tuple[int, ...] | None = None
    use_mesh_loss: bool = False
    use_pose_prior: bool = True
    use_shape_prior: bool = False
    use_limb_loss: bool = False
    learning_rate: float = 0.02
    progressive_keypoint_exposure: bool = False

    @property
    def total_steps(self) -> int:
        return self.iterations * self.steps_per_iter


BODY25 = tuple(range(25))
POSE_INIT_ID_0 = (0, 1, 2, 5, 8, 9, 12)
POSE_INIT_ID_1 = (0, 1, 2, 3, 5, 6, 8, 9, 12, 10, 13, 17, 18)
POSE_INIT_ID_2 = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18)
BODY_CONTOUR = tuple(list(range(25)) + list(range(67, 84)))
HANDS = tuple(range(25, 25 + 21 * 2))
LOWER_ARMS_AND_HANDS = tuple([4, 7] + list(HANDS))
BODY25_PARTS = (
    (1, 8),
    (9, 10),
    (10, 11),
    (8, 9),
    (8, 12),
    (12, 13),
    (13, 14),
    (1, 2),
    (2, 3),
    (3, 4),
    (2, 17),
    (1, 5),
    (5, 6),
    (6, 7),
    (5, 18),
    (1, 0),
    (0, 15),
    (0, 16),
    (15, 17),
    (16, 18),
    (14, 19),
    (19, 20),
    (14, 21),
    (11, 22),
    (22, 23),
    (11, 24),
)


def legacy_schedule(model_type: str, refine_lower_arms: bool = True) -> list[FitStage]:
    """Return the original high-level optimization schedule for a model type."""
    if model_type == "smpl":
        return [
            FitStage("pose_only", 35, 30, ("transl", "global_orient", "body_pose"), BODY25, False, True, False, False, 0.05, True),
            FitStage("pose_shape", 5, 30, ("transl", "global_orient", "body_pose", "betas"), BODY_CONTOUR, True, True, True, False, 0.02),
            FitStage("shape", 3, 30, ("transl", "global_orient", "body_pose", "betas"), None, True, True, True, True, 0.02),
            FitStage("body_pose", 5, 30, ("body_pose",), None, True, True, False, False, 0.02),
        ]

    if model_type == "smplh":
        stages = [
            FitStage("pose_only", 35, 30, ("transl", "global_orient", "body_pose"), BODY25, False, True, False, False, 0.05, True),
            FitStage("pose_shape", 5, 30, ("transl", "global_orient", "body_pose", "betas"), BODY_CONTOUR, True, True, True, False, 0.02),
            FitStage("shape", 3, 30, ("transl", "global_orient", "body_pose", "betas"), None, True, True, True, True, 0.02),
            FitStage("body_pose", 5, 30, ("body_pose",), None, True, True, False, False, 0.02),
        ]
        if refine_lower_arms:
            stages.append(FitStage("lower_arms", 5, 30, ("body_pose",), LOWER_ARMS_AND_HANDS, False, False, False, False, 0.05))
        stages.append(FitStage("hands", 3, 30, ("left_hand_pose", "right_hand_pose"), HANDS, False, False, False, False, 0.05))
        return stages

    if model_type == "smplx":
        stages = [
            FitStage("pose_only", 20, 30, ("transl", "global_orient", "body_pose"), BODY25, False, True, False, False, 0.05, True),
            FitStage("pose_shape", 5, 30, ("transl", "global_orient", "body_pose", "betas"), BODY_CONTOUR, True, True, True, False, 0.02),
            FitStage("shape", 5, 30, ("transl", "global_orient", "body_pose", "betas"), None, True, True, True, True, 0.02),
            FitStage("body_pose", 10, 30, ("body_pose",), None, True, True, False, False, 0.02),
        ]
        if refine_lower_arms:
            stages.append(FitStage("lower_arms", 5, 30, ("body_pose",), LOWER_ARMS_AND_HANDS, False, False, False, False, 0.05))
        stages.extend(
            [
                FitStage("hands", 10, 30, ("left_hand_pose", "right_hand_pose"), HANDS, False, False, False, False, 0.05),
                FitStage("shape", 5, 30, ("transl", "global_orient", "body_pose", "betas"), None, True, True, True, True, 0.02),
                FitStage("body_pose", 10, 30, ("body_pose",), None, True, True, False, False, 0.02),
            ]
        )
        if refine_lower_arms:
            stages.append(FitStage("lower_arms", 5, 30, ("body_pose",), LOWER_ARMS_AND_HANDS, False, False, False, False, 0.05))
        stages.extend(
            [
                FitStage("hands", 10, 30, ("left_hand_pose", "right_hand_pose"), HANDS, False, False, False, False, 0.05),
                FitStage(
                    "face",
                    5,
                    30,
                    ("jaw_pose", "expression", "leye_pose", "reye_pose"),
                    None,
                    False,
                    False,
                    False,
                    False,
                    0.01,
                ),
            ]
        )
        return stages

    raise ValueError(f"Unsupported model type: {model_type}")
