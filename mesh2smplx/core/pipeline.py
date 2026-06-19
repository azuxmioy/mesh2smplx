"""Orchestration layer for mesh, keypoint, and fitting stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .config import PipelineConfig
from .frame_selection import parse_frame_range

if TYPE_CHECKING:
    from .data.interfaces import DataSource


@dataclass
class Pipeline:
    config: PipelineConfig

    def plan(self) -> list[str]:
        stages = [
            f"input: {self.config.input.mode}",
            "normalize cameras and frame ids",
        ]
        if self.config.input.mode == "textured_mesh":
            if self.config.input.images is not None:
                stages.append("load calibrated camera images")
            elif (
                self.config.rendering is not None
                and self.config.rendering.mode == "real"
            ):
                stages.append("render mesh images from calibration cameras")
            elif (
                self.config.rendering is not None
                and self.config.rendering.mode == "virtual"
            ):
                stages.append("render mesh images from heuristic semi-sphere cameras")
            elif self.config.input.cameras is not None:
                stages.append("render mesh images from calibration cameras")
            else:
                stages.append("render mesh images from heuristic semi-sphere cameras")
        stages.extend(
            [
                f"keypoints: {self.config.keypoints.provider}",
                "triangulate 3D keypoints",
                f"fit body model: {self.config.body_model.type}",
                f"write outputs: {self.config.fitting.output_dir}",
            ]
        )
        if self.config.viewer.enabled:
            stages.append("open AITviewer")
        return stages

    def build_source(self) -> "DataSource":
        if self.config.input.mode == "textured_mesh":
            if self.config.input.images is None and self.config.rendering is None:
                raise ValueError("textured_mesh mode requires rendering config")
            from .data.textured_mesh import TexturedMeshSource

            return TexturedMeshSource(self.config.input, self.config.rendering)
        raise ValueError(f"Unsupported input mode: {self.config.input.mode}")

    def run(self) -> None:
        import numpy as np

        from mesh2smplx.openpose.format import load_frame_keypoints
        from mesh2smplx.openpose.providers import build_keypoint_provider
        from .triangulation import triangulate_frame

        frames = parse_frame_range(self.config.fitting.frames)
        source = self.build_source()
        bundle = source.load(frames=frames)

        # 2D keypoint detection (writes per-camera/per-frame OpenPose JSON).
        keypoint_config = self.config.keypoints
        provider = build_keypoint_provider(keypoint_config)
        provider.run(bundle)

        keypoints_dir = keypoint_config.output_dir or keypoint_config.path
        if keypoints_dir is None:
            raise ValueError(
                "Triangulation needs input.keypoints_2d to locate the 2D keypoint JSON files."
            )

        # Stable camera order shared by projection matrices and 2D stacks.
        camera_ids = sorted(bundle.cameras)
        cameras = [bundle.cameras[camera_id] for camera_id in camera_ids]

        keypoints_3d = []
        for frame in bundle.frames:
            keypoints_2d = load_frame_keypoints(
                keypoints_dir,
                camera_ids,
                frame.frame_id,
                thres=keypoint_config.confidence_threshold,
            )
            points_3d, reproj_error = triangulate_frame(cameras, keypoints_2d)
            keypoints_3d.append(points_3d)
            print(f"frame {frame.frame_id}: mean_reprojection_error={reproj_error:.3f}px")

        keypoints_3d = np.stack(keypoints_3d, axis=0)  # (num_frames, num_joints, 5)

        keypoints_path = self.config.input.keypoints_3d or self.config.input.root / "keypoints_3d.npy"
        keypoints_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(keypoints_path, keypoints_3d)

        print(f"frames={bundle.frame_ids}")
        print(f"cameras={camera_ids}")
        print(f"keypoints_3d shape={keypoints_3d.shape}")
        print(f"wrote={keypoints_path}")
        print(
            "Next step: fit with "
            "`python -m mesh2smplx.main --config <config>`"
        )
