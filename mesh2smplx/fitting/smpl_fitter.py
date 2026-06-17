"""SMPL/SMPL-X fitting implementation.

This file intentionally avoids importing heavy optional dependencies at module
import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from mesh2smplx.core.config import BodyModelConfig, FittingConfig
from .joints import JointMapper, smpl_to_openpose
from .losses import (
    body_pose_prior,
    keypoint_objective,
    loss_weights,
    original_joint_weights,
    weighted_keypoint_mse,
)
from .mesh_losses import MeshTarget, symmetric_chamfer_loss
from .schedule import (
    BODY25_PARTS,
    POSE_INIT_ID_0,
    POSE_INIT_ID_1,
    POSE_INIT_ID_2,
    BODY_CONTOUR,
    HANDS,
    FitStage,
    legacy_schedule,
)


@dataclass
class SmplFitResult:
    params: dict[str, torch.Tensor]
    model_type: str
    gender: str
    vertices: torch.Tensor | None = None
    joints: torch.Tensor | None = None
    faces: Any | None = None
    loss: float | None = None


@dataclass
class SmplFitProgress:
    step: int
    total_steps: int
    phase: str
    loss: float
    vertices: torch.Tensor
    joints: torch.Tensor
    target_joints: torch.Tensor
    faces: Any


FitProgressCallback = Callable[[SmplFitProgress], None]


ROOT_ORIENTATION_JOINTS = (1, 2, 5, 8, 9, 12)


@dataclass
class SmplFitter:
    body_config: BodyModelConfig
    fitting_config: FittingConfig

    def __post_init__(self) -> None:
        # Pre-calibrated shape (betas); when set, betas are held fixed.
        self.calibrated_betas = self._resolve_calibrated_betas()
        if self.calibrated_betas is not None:
            print(f"shape calibration: betas fixed ({self.calibrated_betas.shape[-1]} coeffs)")

    def _resolve_calibrated_betas(self) -> "torch.Tensor | None":
        import numpy as np

        cfg = self.body_config
        values = None
        if getattr(cfg, "betas_path", None) is not None:
            path = Path(cfg.betas_path)
            if path.suffix == ".npz":
                values = np.load(path)["betas"]
            elif path.suffix == ".npy":
                values = np.load(path)
            elif path.suffix == ".json":
                import json

                data = json.loads(path.read_text())
                values = data["betas"] if isinstance(data, dict) else data
            else:
                raise ValueError(f"Unsupported betas_path format: {path}")
        elif getattr(cfg, "betas", None):
            values = cfg.betas
        if values is None:
            return None
        tensor = torch.as_tensor(np.asarray(values, dtype=np.float32)).reshape(1, -1)
        return tensor[:, : cfg.num_betas]

    def create_body_model(self, batch_size: int):
        try:
            import smplx
        except ImportError as exc:
            raise RuntimeError("Install the `smplx` package to fit body models.") from exc

        model_type = self.body_config.type
        use_hands = self.body_config.use_hands and model_type in {"smplh", "smplx"}
        use_face = self.body_config.use_face and model_type == "smplx"
        joint_mapper = JointMapper(
            smpl_to_openpose(
                model_type=model_type,
                use_hands=use_hands,
                use_face=use_face,
                use_face_contour=use_face,
            )
        )

        model_kwargs: dict[str, Any] = {
            "model_path": str(self.body_config.model_path),
            "model_type": model_type,
            "gender": self.body_config.gender,
            "joint_mapper": joint_mapper,
            "num_betas": self.body_config.num_betas,
            "batch_size": batch_size,
            "create_global_orient": True,
            "create_transl": True,
            "create_body_pose": True,
            "create_betas": True,
        }
        if model_type in {"smplh", "smplx"}:
            model_kwargs.update(
                {
                    "create_left_hand_pose": use_hands,
                    "create_right_hand_pose": use_hands,
                    "use_pca": True,
                    "num_pca_comps": self.body_config.num_pca_comps,
                    "flat_hand_mean": True,
                }
            )
        if model_type == "smplx":
            model_kwargs.update(
                {
                    "create_expression": use_face,
                    "create_jaw_pose": use_face,
                    "create_leye_pose": use_face,
                    "create_reye_pose": use_face,
                    "num_expression_coeffs": self.body_config.num_expression_coeffs,
                    "use_face_contour": use_face,
                }
            )

        body_model = smplx.create(**model_kwargs).to(self.fitting_config.device)
        if getattr(self, "calibrated_betas", None) is not None and hasattr(body_model, "betas"):
            with torch.no_grad():
                num = body_model.betas.shape[1]
                body_model.betas.data[:] = self.calibrated_betas[:, :num].to(body_model.betas.device)
        return body_model

    @property
    def _betas_calibrated(self) -> bool:
        return getattr(self, "calibrated_betas", None) is not None

    def _load_params(self, body_model, params: dict) -> None:
        """Warm-start: copy a previous frame's parameters into the model."""
        device = next(body_model.parameters()).device
        with torch.no_grad():
            for name, value in params.items():
                if not hasattr(body_model, name):
                    continue
                target = getattr(body_model, name)
                src = torch.as_tensor(value, dtype=target.dtype, device=device)
                if src.shape == target.shape:
                    target.data.copy_(src)
                elif src.shape[1:] == target.shape[1:]:
                    target.data.copy_(src[: target.shape[0]])

    def fit_keypoints(
        self,
        keypoints_3d: torch.Tensor,
        iterations: int | None = None,
        learning_rate: float | None = None,
        steps_per_iter: int | None = None,
        pose_shape_iters: int | None = None,
        hand_iters: int | None = None,
        face_iters: int | None = None,
        optimize_shape: bool = True,
        staged: bool = True,  # kept for back-compat; the keypoint schedule is always staged
        log_every: int = 25,
        progress_callback: FitProgressCallback | None = None,
        callback_interval: int = 25,
    ) -> SmplFitResult:
        """Fit body model to 3D keypoints, reproducing the legacy keypoint schedule.

        This mirrors the original ``BaseFitter.optimize_pose_only`` followed by
        the keypoint terms of ``optimize_pose_shape`` (the scan/ICP terms are
        dropped — this is the keypoint-only path): progressive joint exposure,
        the per-joint ``body_pose_prior`` exp barriers, legacy joint weights, and
        the ``/(1+it)`` loss-weight schedule (pose-only uses ``it/4``).

        ``iterations`` is the number of OUTER iterations of the pose-only stage
        (each runs ``steps_per_iter`` Adam steps); ``pose_shape_iters`` outer
        iterations of the betas-enabled stage follow.
        """
        device = torch.device(self.fitting_config.device)
        keypoints_3d = keypoints_3d.to(device=device, dtype=torch.float32)
        body_model = self.create_body_model(batch_size=len(keypoints_3d))
        self._initialize_root_orientation_from_keypoints(body_model, keypoints_3d)
        self._initialize_translation_from_keypoints(body_model, keypoints_3d)

        # Resolve the schedule from config; explicit method args override it.
        sched = self.fitting_config.schedule
        iterations = int(sched["iterations"]) if iterations is None else iterations
        steps_per_iter = int(sched["steps_per_iter"]) if steps_per_iter is None else steps_per_iter
        pose_shape_iters = int(sched["pose_shape_iters"]) if pose_shape_iters is None else pose_shape_iters
        hand_iters = int(sched["hand_iters"]) if hand_iters is None else hand_iters
        face_iters = int(sched["face_iters"]) if face_iters is None else face_iters
        pose_only_lr = float(sched["pose_only_lr"]) if learning_rate is None else learning_rate
        pose_shape_lr = float(sched["pose_shape_lr"])
        hand_lr = float(sched["hand_lr"])
        face_lr = float(sched["face_lr"])

        num_joints = keypoints_3d.shape[1]
        with torch.no_grad():
            upper = min(body_model(return_verts=False).joints.shape[1], num_joints)
        joint_weights = original_joint_weights(num_joints, device=device)
        weight = loss_weights(self.fitting_config.loss_weights)

        body_contour = torch.tensor(
            [idx for idx in BODY_CONTOUR if idx < upper], dtype=torch.long, device=device
        )
        hand_ids = torch.tensor([idx for idx in HANDS if idx < upper], dtype=torch.long, device=device)
        all_ids = torch.arange(upper, dtype=torch.long, device=device)

        pose_params = [body_model.transl, body_model.global_orient, body_model.body_pose]
        shape_params = list(pose_params)
        if optimize_shape and hasattr(body_model, "betas") and not self._betas_calibrated:
            shape_params = pose_params + [body_model.betas]

        # Each stage: name, params, outer_iters, lr, joint-id tensor (None=progressive),
        # term names to compute, and the it-schedule divisor.
        stages: list[tuple] = [
            ("pose_only", pose_params, iterations, pose_only_lr, None, ("pose_obj", "pose_pr"), 4.0),
            ("pose_shape", shape_params, pose_shape_iters, pose_shape_lr, body_contour, ("pose_obj", "pose_pr", "betas"), 1.0),
        ]
        # refine_hands: optimize the hand PCA against the hand keypoints (25:67).
        if hand_iters > 0 and self.body_config.use_hands and hasattr(body_model, "left_hand_pose") and len(hand_ids) > 0:
            stages.append(
                ("hands", [body_model.left_hand_pose, body_model.right_hand_pose],
                 hand_iters, hand_lr, hand_ids, ("pose_obj",), 1.0)
            )
        # refine_face: optimize jaw/expression/eyes against all keypoints.
        if face_iters > 0 and self.body_config.use_face and hasattr(body_model, "jaw_pose"):
            face_params = [body_model.jaw_pose, body_model.expression,
                           body_model.leye_pose, body_model.reye_pose]
            stages.append(("face", face_params, face_iters, face_lr, all_ids, ("pose_obj", "jaw", "f_exp"), 1.0))

        total_steps = sum(stage[2] for stage in stages) * steps_per_iter
        total_step = 0
        last_loss = None
        for name, params, outer_iters, lr, stage_ids, terms, sched_scale in stages:
            optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
            for it in range(outer_iters):
                ids = self._progressive_ids(it, upper, device) if stage_ids is None else stage_ids
                sched_it = it / sched_scale
                for _ in range(steps_per_iter):
                    total_step += 1
                    optimizer.zero_grad(set_to_none=True)
                    output = body_model(return_verts=False)
                    raw: dict[str, torch.Tensor] = {
                        "pose_obj": keypoint_objective(
                            keypoints_3d[:, ids], output.joints[:, ids], joint_weights[ids]
                        )
                    }
                    if "pose_pr" in terms and hasattr(body_model, "body_pose"):
                        raw["pose_pr"] = body_pose_prior(body_model.body_pose)
                    if "betas" in terms and hasattr(body_model, "betas") and not self._betas_calibrated:
                        raw["betas"] = (body_model.betas ** 2).mean()
                    if "jaw" in terms and hasattr(body_model, "jaw_pose"):
                        raw["jaw"] = (body_model.jaw_pose ** 2).sum()
                    if "f_exp" in terms and hasattr(body_model, "expression"):
                        raw["f_exp"] = (body_model.expression ** 2).sum()
                    weighted = {key: weight[key](value, sched_it) for key, value in raw.items()}
                    loss = torch.stack(list(weighted.values())).sum()
                    loss.backward()
                    optimizer.step()
                    last_loss = float(loss.detach().cpu())
                    if log_every and (total_step == 1 or total_step % log_every == 0):
                        parts = ", ".join(
                            f"{key}={value.detach().item():.4f}" for key, value in weighted.items()
                        )
                        print(
                            f"step {total_step:04d}/{total_steps} stage={name} it={it:02d} "
                            f"njoints={len(ids)} loss={last_loss:.4f} {parts}"
                        )
                    if progress_callback is not None and self._should_emit_progress(
                        total_step, total_steps, callback_interval
                    ):
                        with torch.no_grad():
                            snapshot = body_model(return_verts=True)
                            progress_callback(
                                SmplFitProgress(
                                    step=total_step,
                                    total_steps=total_steps,
                                    phase=name,
                                    loss=last_loss,
                                    vertices=snapshot.vertices.detach().cpu(),
                                    joints=snapshot.joints.detach().cpu(),
                                    target_joints=keypoints_3d[..., :3].detach().cpu(),
                                    faces=body_model.faces,
                                )
                            )

        with torch.no_grad():
            output = body_model(return_verts=True)
            params = {key: val.detach().cpu() for key, val in body_model.named_parameters()}
            vertices = output.vertices.detach().cpu()
            joints = output.joints.detach().cpu()

        return SmplFitResult(
            params=params,
            model_type=self.body_config.type,
            gender=self.body_config.gender,
            vertices=vertices,
            joints=joints,
            faces=body_model.faces,
            loss=last_loss,
        )

    def fit_full(
        self,
        keypoints_3d: torch.Tensor,
        mesh_targets: list[MeshTarget] | None = None,
        schedule: list[FitStage] | None = None,
        progress_callback: FitProgressCallback | None = None,
        callback_interval: int = 25,
        max_steps_per_stage: int | None = None,
        max_total_steps: int | None = None,
        body_vertex_samples: int = 5000,
        init_params: dict | None = None,
    ) -> SmplFitResult:
        """Run the staged mesh-aware fitting pipeline (batched over frames).

        3D keypoints initialize and constrain the fit, while mesh stages add
        scan/body alignment. When ``init_params`` is given, the model is
        warm-started from those parameters and the keypoint-based root/translation
        init is skipped (used by tracking). Pure PyTorch nearest-neighbour mesh
        loss keeps the core path free of PyTorch3D/Chumpy/OpenDR/psbody.
        """
        device = torch.device(self.fitting_config.device)
        keypoints_3d = keypoints_3d.to(device=device, dtype=torch.float32)
        body_model = self.create_body_model(batch_size=len(keypoints_3d))
        if init_params is not None:
            self._load_params(body_model, init_params)
        else:
            self._initialize_root_orientation_from_keypoints(body_model, keypoints_3d)
            self._initialize_translation_from_keypoints(body_model, keypoints_3d)
        schedule = schedule or legacy_schedule(self.body_config.type)
        target_points = self._stack_target_points(mesh_targets, len(keypoints_3d), device)
        stage_step_limits = self._stage_step_limits(schedule, max_steps_per_stage, max_total_steps)

        last_loss = self._run_full_schedule(
            body_model, keypoints_3d, target_points, schedule, stage_step_limits,
            body_vertex_samples, progress_callback, callback_interval,
        )
        return self._extract_result(body_model, last_loss)

    def fit_full_sequence(
        self,
        keypoints_3d: torch.Tensor,
        mesh_targets: list[MeshTarget] | None = None,
        tracking: bool = True,
        schedule: list[FitStage] | None = None,
        progress_callback: FitProgressCallback | None = None,
        callback_interval: int = 25,
        max_steps_per_stage: int | None = None,
        max_total_steps: int | None = None,
        body_vertex_samples: int = 5000,
    ) -> list[SmplFitResult]:
        """Fit a sequence of frames one at a time.

        With ``tracking=True`` each frame after the first warm-starts from the
        previous frame's parameters (the model is reused), so the root/translation
        init is only solved for frame 0. The shared model also keeps any
        calibrated betas fixed across the whole sequence.
        """
        device = torch.device(self.fitting_config.device)
        keypoints_3d = keypoints_3d.to(device=device, dtype=torch.float32)
        body_model = self.create_body_model(batch_size=1)
        schedule = schedule or legacy_schedule(self.body_config.type)
        stage_step_limits = self._stage_step_limits(schedule, max_steps_per_stage, max_total_steps)

        results: list[SmplFitResult] = []
        for index in range(len(keypoints_3d)):
            frame_keypoints = keypoints_3d[index : index + 1]
            if index == 0 or not tracking:
                self._initialize_root_orientation_from_keypoints(body_model, frame_keypoints)
                self._initialize_translation_from_keypoints(body_model, frame_keypoints)
            else:
                print(f"frame {index}: warm-start from previous frame (tracking)")
            target = [mesh_targets[index]] if mesh_targets else None
            target_points = self._stack_target_points(target, 1, device)
            last_loss = self._run_full_schedule(
                body_model, frame_keypoints, target_points, schedule, stage_step_limits,
                body_vertex_samples, progress_callback, callback_interval,
            )
            results.append(self._extract_result(body_model, last_loss))
        return results

    def _run_full_schedule(
        self,
        body_model,
        keypoints_3d: torch.Tensor,
        target_points: torch.Tensor | None,
        schedule: list[FitStage],
        stage_step_limits: list[int],
        body_vertex_samples: int,
        progress_callback: FitProgressCallback | None,
        callback_interval: int,
    ) -> float | None:
        total_steps = sum(stage_step_limits)
        last_loss = None
        step = 0
        for stage, stage_steps in zip(schedule, stage_step_limits):
            if stage_steps <= 0:
                continue
            opt_params = self._params_for_stage(body_model, stage)
            if self._betas_calibrated:
                opt_params = [p for p in opt_params if p is not getattr(body_model, "betas", None)]
            if not opt_params:
                print(f"skip stage={stage.name}: no matching parameters for {self.body_config.type}")
                continue

            optimizer = torch.optim.Adam(opt_params, lr=stage.learning_rate, betas=(0.9, 0.999))

            for stage_step in range(stage_steps):
                step += 1
                optimizer.zero_grad(set_to_none=True)
                output = body_model(return_verts=stage.use_mesh_loss)
                loss_dict = self._full_loss_dict(
                    output=output,
                    keypoints_3d=keypoints_3d,
                    stage=stage,
                    target_points=target_points,
                    body_vertex_samples=body_vertex_samples,
                    stage_iteration=stage_step // max(1, stage.steps_per_iter),
                )
                loss = torch.stack(list(loss_dict.values())).sum()
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach().cpu())

                if step == 1 or step % 25 == 0 or step == total_steps:
                    loss_parts = ", ".join(
                        f"{key}={value.detach().item():.5f}" for key, value in loss_dict.items()
                    )
                    print(
                        f"step {step:04d}/{total_steps} stage={stage.name} "
                        f"loss={last_loss:.6f} {loss_parts}"
                    )

                if progress_callback is not None and self._should_emit_progress(
                    step, total_steps, callback_interval
                ):
                    with torch.no_grad():
                        snapshot = body_model(return_verts=True)
                        progress_callback(
                            SmplFitProgress(
                                step=step,
                                total_steps=total_steps,
                                phase=stage.name,
                                loss=last_loss,
                                vertices=snapshot.vertices.detach().cpu(),
                                joints=snapshot.joints.detach().cpu(),
                                target_joints=keypoints_3d[..., :3].detach().cpu(),
                                faces=body_model.faces,
                            )
                        )
        return last_loss

    def _extract_result(self, body_model, last_loss: float | None) -> SmplFitResult:
        with torch.no_grad():
            output = body_model(return_verts=True)
            params = {key: val.detach().cpu() for key, val in body_model.named_parameters()}
            return SmplFitResult(
                params=params,
                model_type=self.body_config.type,
                gender=self.body_config.gender,
                vertices=output.vertices.detach().cpu(),
                joints=output.joints.detach().cpu(),
                faces=body_model.faces,
                loss=last_loss,
            )

    @staticmethod
    def _progressive_ids(outer_it: int, upper: int, device: torch.device) -> torch.Tensor:
        """Legacy progressive joint exposure for the pose-only stage."""
        if outer_it < 5:
            values = POSE_INIT_ID_0
        elif outer_it < 10:
            values = POSE_INIT_ID_1
        elif outer_it < 15:
            values = POSE_INIT_ID_2
        else:
            values = BODY_CONTOUR
        return torch.tensor([idx for idx in values if idx < upper], dtype=torch.long, device=device)

    @staticmethod
    def _joint_weights(joint_count: int, device: torch.device) -> torch.Tensor:
        weights = torch.ones(joint_count, 1, dtype=torch.float32, device=device)
        if joint_count >= 25:
            weights[:25] = 8.0
        if joint_count >= 67:
            weights[25:67] = 1.5
        if joint_count > 67:
            weights[67:] = 0.35
        return weights

    @staticmethod
    def _keypoint_confidence(keypoints_3d: torch.Tensor) -> torch.Tensor:
        if keypoints_3d.shape[-1] > 4:
            return keypoints_3d[..., 4]
        if keypoints_3d.shape[-1] > 3:
            return keypoints_3d[..., 3]
        return torch.ones(keypoints_3d.shape[:-1], dtype=keypoints_3d.dtype, device=keypoints_3d.device)

    def _initialize_root_orientation_from_keypoints(
        self,
        body_model: Any,
        keypoints_3d: torch.Tensor,
        joint_indices: tuple[int, ...] = ROOT_ORIENTATION_JOINTS,
    ) -> None:
        """Initialize global orientation from torso keypoints before local pose fitting."""
        if not hasattr(body_model, "global_orient"):
            return

        with torch.no_grad():
            output = body_model(return_verts=False)
            upper = min(output.joints.shape[1], keypoints_3d.shape[1])
            indices = torch.tensor(
                [idx for idx in joint_indices if idx < upper],
                dtype=torch.long,
                device=keypoints_3d.device,
            )
            if indices.numel() < 3:
                return

            source = output.joints[:, indices]
            target = keypoints_3d[:, indices, :3]
            confidence = self._keypoint_confidence(keypoints_3d[:, indices]).clamp_min(0.0)
            valid = torch.isfinite(target).all(dim=-1) & torch.isfinite(confidence) & (confidence > 0)

            rotations = []
            initialized = []
            identity = torch.eye(3, dtype=source.dtype, device=source.device)
            for batch_index in range(source.shape[0]):
                sample_valid = valid[batch_index]
                if int(sample_valid.sum().item()) < 3:
                    rotations.append(identity)
                    initialized.append(False)
                    continue
                rotation = self._weighted_kabsch_rotation(
                    source[batch_index, sample_valid],
                    target[batch_index, sample_valid],
                    confidence[batch_index, sample_valid],
                )
                rotations.append(rotation)
                initialized.append(True)

            if not any(initialized):
                return

            rotation_mats = torch.stack(rotations, dim=0)
            root_orient = self._matrix_to_axis_angle(rotation_mats)
            current = body_model.global_orient.detach().clone()
            initialized_mask = torch.tensor(initialized, dtype=torch.bool, device=keypoints_3d.device)
            current[initialized_mask] = root_orient[initialized_mask]
            body_model.global_orient.copy_(current)

            angles_deg = torch.linalg.norm(root_orient[initialized_mask], dim=-1) * (180.0 / torch.pi)
            mean_angle = float(angles_deg.mean().detach().cpu())
            print(f"initialized global_orient from keypoints mean_angle_deg={mean_angle:.2f}")

    @staticmethod
    def _weighted_kabsch_rotation(
        source: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        weights = weights / weights.sum().clamp_min(1e-8)
        source_center = (source * weights[:, None]).sum(dim=0, keepdim=True)
        target_center = (target * weights[:, None]).sum(dim=0, keepdim=True)
        source_centered = source - source_center
        target_centered = target - target_center
        covariance = source_centered.transpose(0, 1) @ (target_centered * weights[:, None])
        u, _, vh = torch.linalg.svd(covariance)
        v = vh.transpose(-1, -2)
        rotation = v @ u.transpose(-1, -2)
        if torch.det(rotation) < 0:
            v = v.clone()
            v[:, -1] *= -1
            rotation = v @ u.transpose(-1, -2)
        return rotation

    @staticmethod
    def _matrix_to_axis_angle(rotation_mats: torch.Tensor) -> torch.Tensor:
        trace = torch.diagonal(rotation_mats, dim1=-2, dim2=-1).sum(dim=-1)
        cos_angle = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        angle = torch.acos(cos_angle)
        axis = torch.stack(
            (
                rotation_mats[:, 2, 1] - rotation_mats[:, 1, 2],
                rotation_mats[:, 0, 2] - rotation_mats[:, 2, 0],
                rotation_mats[:, 1, 0] - rotation_mats[:, 0, 1],
            ),
            dim=-1,
        )
        axis = axis / (2.0 * torch.sin(angle).unsqueeze(-1)).clamp_min(1e-8)
        axis_angle = axis * angle.unsqueeze(-1)
        return torch.where(angle.unsqueeze(-1) < 1e-6, torch.zeros_like(axis_angle), axis_angle)

    def _initialize_translation_from_keypoints(
        self,
        body_model: Any,
        keypoints_3d: torch.Tensor,
        joint_indices: tuple[int, ...] = POSE_INIT_ID_0,
    ) -> None:
        """Place the model near the observed skeleton before pose optimization."""
        if not hasattr(body_model, "transl"):
            return

        with torch.no_grad():
            output = body_model(return_verts=False)
            upper = min(output.joints.shape[1], keypoints_3d.shape[1])
            indices = torch.tensor(
                [idx for idx in joint_indices if idx < upper],
                dtype=torch.long,
                device=keypoints_3d.device,
            )
            if indices.numel() == 0:
                return

            target = keypoints_3d[:, indices, :3]
            predicted = output.joints[:, indices]
            confidence = self._keypoint_confidence(keypoints_3d[:, indices]).clamp_min(0.0)
            valid = torch.isfinite(target).all(dim=-1) & torch.isfinite(confidence) & (confidence > 0)
            confidence = torch.where(valid, confidence, torch.zeros_like(confidence))
            denominator = confidence.sum(dim=1, keepdim=True).clamp_min(1e-8)
            delta = torch.where(valid.unsqueeze(-1), target - predicted, torch.zeros_like(target))
            translation_delta = (delta * confidence.unsqueeze(-1)).sum(dim=1) / denominator
            body_model.transl.add_(translation_delta)
            mean_delta_mm = float(torch.linalg.norm(translation_delta, dim=-1).mean().detach().cpu() * 1000.0)
            print(f"initialized transl from keypoints mean_delta_mm={mean_delta_mm:.2f}")

    def _full_loss_dict(
        self,
        output: Any,
        keypoints_3d: torch.Tensor,
        stage: FitStage,
        target_points: torch.Tensor | None,
        body_vertex_samples: int,
        stage_iteration: int,
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        weight = loss_weights(self.fitting_config.loss_weights)
        joints = output.joints
        joint_indices = self._stage_joint_indices(
            stage,
            joints.shape[1],
            keypoints_3d.shape[1],
            joints.device,
            stage_iteration=stage_iteration,
        )
        if len(joint_indices) > 0:
            jw = original_joint_weights(keypoints_3d.shape[1], device=joints.device)[joint_indices]
            keypoint_loss = keypoint_objective(
                keypoints_3d[:, joint_indices], joints[:, joint_indices], jw
            )
            losses["pose_obj"] = weight["pose_obj"](keypoint_loss, stage_iteration)

        if stage.use_mesh_loss and target_points is not None:
            chamfer = symmetric_chamfer_loss(
                output.vertices, target_points, vertex_samples=body_vertex_samples
            )
            losses["icp"] = weight["icp"](chamfer, stage_iteration)

        if stage.use_pose_prior and hasattr(output, "body_pose"):
            losses["pose_pr"] = weight["pose_pr"](body_pose_prior(output.body_pose), stage_iteration)

        if stage.use_shape_prior and hasattr(output, "betas") and not self._betas_calibrated:
            losses["betas"] = weight["betas"](output.betas.square().mean(), stage_iteration)

        if stage.use_limb_loss and joints.shape[1] >= 25 and keypoints_3d.shape[1] >= 25:
            edge_index = torch.tensor(BODY25_PARTS, dtype=torch.long, device=joints.device)
            observed = torch.linalg.norm(
                keypoints_3d[:, edge_index[:, 1], :3] - keypoints_3d[:, edge_index[:, 0], :3],
                dim=-1,
            )
            predicted = torch.linalg.norm(
                joints[:, edge_index[:, 1]] - joints[:, edge_index[:, 0]],
                dim=-1,
            )
            losses["limb"] = weight["limb"]((observed - predicted).square().mean(), stage_iteration)

        return losses

    @staticmethod
    def _stage_joint_indices(
        stage: FitStage,
        model_joint_count: int,
        target_joint_count: int,
        device: torch.device,
        stage_iteration: int = 0,
    ) -> torch.Tensor:
        upper = min(model_joint_count, target_joint_count)
        if stage.progressive_keypoint_exposure:
            if stage_iteration < 5:
                values = list(POSE_INIT_ID_0)
            elif stage_iteration < 10:
                values = list(POSE_INIT_ID_1)
            elif stage_iteration < 15:
                values = list(POSE_INIT_ID_2)
            else:
                values = list(BODY_CONTOUR)
        elif stage.joint_slice is None:
            values = list(range(upper))
        elif isinstance(stage.joint_slice, slice):
            values = list(range(upper))[stage.joint_slice]
        else:
            values = [idx for idx in stage.joint_slice if idx < upper]
        values = [idx for idx in values if idx < upper]
        return torch.tensor(values, dtype=torch.long, device=device)

    @staticmethod
    def _stage_step_limits(
        schedule: list[FitStage],
        max_steps_per_stage: int | None,
        max_total_steps: int | None,
    ) -> list[int]:
        limits = [
            min(stage.total_steps, max_steps_per_stage)
            if max_steps_per_stage is not None
            else stage.total_steps
            for stage in schedule
        ]
        if max_total_steps is None:
            return limits
        remaining = max(0, max_total_steps)
        capped = []
        for limit in limits:
            stage_limit = min(limit, remaining)
            capped.append(stage_limit)
            remaining -= stage_limit
        return capped

    @staticmethod
    def _params_for_stage(body_model: Any, stage: FitStage) -> list[torch.Tensor]:
        params = []
        for name in stage.optimize:
            if hasattr(body_model, name):
                params.append(getattr(body_model, name))
        return params

    @staticmethod
    def _stack_target_points(
        mesh_targets: list[MeshTarget] | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if not mesh_targets:
            return None
        points = torch.cat([target.points.to(device=device) for target in mesh_targets], dim=0)
        if len(mesh_targets) == 1 and batch_size > 1:
            points = points.repeat(batch_size, 1, 1)
        if points.shape[0] != batch_size:
            raise ValueError(
                f"Expected {batch_size} mesh targets, got {points.shape[0]} point batches"
            )
        return points

    @staticmethod
    def _staged_phases(
        iterations: int,
        learning_rate: float,
        body_model: Any,
        body_params: list[torch.Tensor],
        full_params: list[torch.Tensor],
    ):
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if iterations < 3:
            return [("full", full_params, iterations, learning_rate, slice(None))]

        stage1 = max(1, iterations // 5)
        stage2 = max(1, iterations // 3)
        stage3 = max(1, iterations - stage1 - stage2)

        while stage1 + stage2 + stage3 > iterations:
            if stage3 > 1:
                stage3 -= 1
            elif stage2 > 1:
                stage2 -= 1
            else:
                stage1 -= 1

        return [
            (
                "global_body",
                [body_model.global_orient, body_model.transl],
                stage1,
                learning_rate,
                slice(0, 25),
            ),
            ("body", body_params, stage2, learning_rate * 0.7, slice(0, 25)),
            ("full", full_params, stage3, learning_rate * 0.5, slice(None)),
        ]

    @staticmethod
    def _should_emit_progress(step: int, total_steps: int, interval: int) -> bool:
        interval = max(1, interval)
        return step == 1 or step == total_steps or step % interval == 0

    def save_result(self, result: SmplFitResult, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        serializable: dict[str, Any] = {
            "model_type": result.model_type,
            "gender": result.gender,
            "params": {key: value.detach().cpu().tolist() for key, value in result.params.items()},
        }
        import json

        with open(output_dir / "smpl_params.json", "w", encoding="utf-8") as handle:
            json.dump(serializable, handle, indent=2)

    def save_mesh_obj(self, result: SmplFitResult, output_path: Path, batch_index: int = 0) -> None:
        if result.vertices is None or result.faces is None:
            raise ValueError("Fit result does not contain vertices/faces")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        vertices = result.vertices[batch_index].numpy()
        faces = result.faces
        with open(output_path, "w", encoding="utf-8") as handle:
            for vertex in vertices:
                handle.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
            for face in faces:
                handle.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")
