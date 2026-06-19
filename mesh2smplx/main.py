"""Single-purpose command line entry point for SMPL-family mesh fitting."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from .core.config import load_config
from .core.frame_selection import parse_frame_range
from .core.pipeline import Pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mesh2smplx",
        description=(
            "Fit a SMPL-family body model to a mesh sequence. If 3D keypoints are "
            "missing, they are generated from the configured images or rendered mesh views first."
        ),
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to a YAML config.")
    parser.add_argument(
        "--keypoints3d",
        type=Path,
        default=None,
        help="Path to keypoints .npy. Defaults to input.keypoints_3d / data/keypoints_3d.npy.",
    )
    parser.add_argument(
        "--frame-indices",
        default=None,
        help=(
            "Optional index list/range into the sorted mesh sequence, for example '0,2-4'. "
            "Omit to fit every mesh in order."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory.")
    parser.add_argument(
        "--betas",
        type=Path,
        default=None,
        help=(
            "Shape calibration: path to betas (.npy/.npz with `betas`/.json). Overrides "
            "input.body_shape/config betas; betas are held fixed."
        ),
    )
    parser.add_argument(
        "--aitviewer-remote",
        default=None,
        help="Optional AITviewer remote server as HOST:PORT, for example localhost:8417.",
    )
    parser.add_argument("--aitviewer-launch", action="store_true", help="Launch a local AITviewer server.")
    parser.add_argument("--aitviewer-update-interval", type=int, default=25)
    parser.add_argument("--aitviewer-timeout", type=float, default=10.0)
    parser.add_argument("--aitviewer-log", type=Path, default=None)
    parser.add_argument(
        "--aitviewer-window-type",
        default=None,
        help="Override AITviewer backend, for example pyqt6, pyqt5, or glfw.",
    )
    parser.add_argument(
        "--aitviewer-camera-overlay",
        action="store_true",
        help="When launching AITviewer, add calibrated cameras and image billboards.",
    )
    parser.add_argument(
        "--aitviewer-camera-json",
        type=Path,
        default=None,
        help="Override camera calibration JSON. Defaults to input.cameras.",
    )
    parser.add_argument(
        "--aitviewer-image-root",
        type=Path,
        default=None,
        help="Override camera image root. Defaults to input.images.",
    )
    parser.add_argument(
        "--aitviewer-cameras",
        default=None,
        help="Comma-separated camera ids to show. Defaults to the first --aitviewer-max-cameras ids.",
    )
    parser.add_argument("--aitviewer-max-cameras", type=int, default=4)
    parser.add_argument("--aitviewer-camera-scale", type=float, default=None)
    parser.add_argument(
        "--aitviewer-initial-camera",
        default="auto",
        help=(
            "Initial camera for launched AITviewer windows: 'auto' uses the first "
            "configured camera, 'none' disables it, or pass a camera id."
        ),
    )
    parser.add_argument("--aitviewer-billboard-distance", type=float, default=2.0)
    parser.add_argument("--aitviewer-billboard-alpha", type=float, default=0.55)
    parser.add_argument("--aitviewer-image-extensions", default=".png,.jpg,.jpeg")
    parser.add_argument("--aitviewer-shadows", action="store_true", help="Enable viewer shadows.")
    parser.add_argument("--aitviewer-znear", type=float, default=0.05)
    parser.add_argument("--aitviewer-zfar", type=float, default=50.0)
    return parser


def _write_fit_results(
    fitter,
    results,
    mesh_frames,
    mesh_targets,
    output_dir: Path,
    tracking: bool,
) -> float | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stems = _mesh_output_stems(mesh_frames)
    if tracking and len(mesh_frames) > 1:
        # One result per frame (each batch size 1).
        for result, mesh_frame, mesh_target, stem in zip(
            results, mesh_frames, mesh_targets, output_stems
        ):
            _write_one_frame_result(
                fitter, result, mesh_frame, mesh_target, stem, output_dir, batch_index=0
            )
        return results[-1].loss

    # One batched result covering all selected frames.
    result = results[0]
    for batch_index, (mesh_frame, mesh_target, stem) in enumerate(
        zip(mesh_frames, mesh_targets, output_stems)
    ):
        _write_one_frame_result(
            fitter, result, mesh_frame, mesh_target, stem, output_dir, batch_index=batch_index
        )
    return result.loss


def _write_one_frame_result(
    fitter,
    result,
    mesh_frame,
    mesh_target,
    stem: str,
    output_dir: Path,
    *,
    batch_index: int,
) -> None:
    model_type = result.model_type.lower()
    metadata = {
        "frame_id": mesh_frame.frame_id,
        "source_mesh": str(mesh_frame.mesh_path),
    }
    fitter.save_result_json(
        result,
        output_dir / f"{stem}_{model_type}_params.json",
        batch_index=batch_index,
        metadata=metadata,
    )
    fitter.save_mesh_obj(
        result,
        output_dir / f"{stem}_{model_type}.obj",
        batch_index=batch_index,
    )
    _write_mesh_target_obj(mesh_target, output_dir / f"{stem}_scan.obj")


def _mesh_output_stems(mesh_frames) -> list[str]:
    raw_stems = [_mesh_output_stem(frame) for frame in mesh_frames]
    duplicate_stems = {stem for stem in raw_stems if raw_stems.count(stem) > 1}
    stems = []
    for frame, stem in zip(mesh_frames, raw_stems):
        if stem in duplicate_stems:
            stem = f"{stem}_{frame.frame_id:06d}"
        stems.append(stem)
    return stems


def _mesh_output_stem(mesh_frame) -> str:
    stem = mesh_frame.mesh_path.stem
    if stem.lower() in {"mesh", "model", "scan"} and mesh_frame.mesh_path.parent.name:
        stem = mesh_frame.mesh_path.parent.name
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return stem or f"{mesh_frame.frame_id:06d}"


def _write_mesh_target_obj(mesh_target, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vertices = mesh_target.vertices[0].detach().cpu().numpy()
    faces = mesh_target.faces.detach().cpu().numpy()
    with open(output_path, "w", encoding="utf-8") as handle:
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        for face in faces:
            handle.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")


def _build_fit_converter(config, target_type: str):
    from .fitting.conversion import (
        SmplModelConverter,
        build_target_body_config,
        infer_transfer_matrix_path,
    )

    transfer_matrix = config.conversion.transfer_matrix
    transfer_dir = config.conversion.transfer_dir
    if transfer_matrix is None:
        transfer_matrix = infer_transfer_matrix_path(
            config.body_model.type,
            target_type,
            transfer_dir,
        )

    target_config = build_target_body_config(
        config.body_model,
        target_type,
        model_path=config.conversion.model_path or config.body_model.model_path,
        gender=config.conversion.gender or config.body_model.gender,
    )
    return SmplModelConverter(
        source_body_config=config.body_model,
        target_body_config=target_config,
        fitting_config=config.fitting,
        transfer_matrix_path=transfer_matrix,
        num_steps=config.conversion.num_steps,
        learning_rate=config.conversion.learning_rate,
        beta_regularizer=config.conversion.beta_regularizer,
        optimize_betas=config.conversion.optimize_betas,
    )


def _resolve_output_model_types(config) -> list[str]:
    output_types = list(config.conversion.output_types)
    if not output_types:
        raise ValueError("At least one output model type is required.")
    return output_types


def _model_output_dir(base_output_dir: Path, model_type: str) -> Path:
    return base_output_dir / model_type


def _write_output_model_results(
    config,
    native_fitter,
    results,
    mesh_frames,
    mesh_targets,
    fit_output_dir: Path,
    tracking,
) -> dict[str, Path]:
    from .fitting import SmplFitter

    output_types = _resolve_output_model_types(config)
    base_output_dir = config.conversion.output_dir or fit_output_dir
    written: dict[str, Path] = {}

    for output_type in output_types:
        output_dir = _model_output_dir(base_output_dir, output_type)
        if output_type == config.body_model.type:
            target_fitter = native_fitter
            output_results = results
        else:
            print(f"converting {config.body_model.type}->{output_type}")
            converter = _build_fit_converter(config, output_type)
            output_results = [converter.convert_result(result) for result in results]
            target_fitter = SmplFitter(converter.target_body_config, config.fitting)
        _write_fit_results(
            target_fitter,
            output_results,
            mesh_frames,
            mesh_targets,
            output_dir,
            tracking=tracking,
        )
        written[output_type] = output_dir

    return written


def _resolve_keypoints_path(config, args: argparse.Namespace) -> Path:
    keypoints_path = args.keypoints3d or config.input.keypoints_3d
    if keypoints_path is None:
        keypoints_path = config.input.root / "keypoints_3d.npy"
    if args.keypoints3d is not None:
        config.input.keypoints_3d = args.keypoints3d
    return keypoints_path


def _ensure_keypoints3d(config, keypoints_path: Path) -> None:
    if keypoints_path.exists():
        return

    print(f"3D keypoints not found at {keypoints_path}; generating them from the config.")
    original_frames = config.fitting.frames
    config.fitting.frames = None
    try:
        Pipeline(config).run()
    finally:
        config.fitting.frames = original_frames

    if not keypoints_path.exists():
        raise FileNotFoundError(
            f"3D keypoint generation finished but did not write {keypoints_path}. "
            "Check input.keypoints_3d and keypoints.output_dir in the config."
        )


def fit_command(args: argparse.Namespace) -> None:
    import numpy as np
    import torch

    from .core.data.mesh_sequence import discover_mesh_sequence
    from .fitting import SmplFitter
    from .fitting.mesh_losses import load_mesh_target

    config = load_config(args.config)
    if args.betas is not None:
        config.body_model.betas_path = args.betas
    tracking = config.fitting.tracking
    scan_surface_samples = config.fitting.scan_surface_samples
    body_vertex_samples = config.fitting.body_vertex_samples
    max_steps_per_stage = config.fitting.max_steps_per_stage
    max_total_steps = config.fitting.max_total_steps

    keypoints_path = _resolve_keypoints_path(config, args)
    _ensure_keypoints3d(config, keypoints_path)

    mesh_frames = discover_mesh_sequence(config.input)
    keypoints = np.load(keypoints_path)
    frame_indices = parse_frame_range(args.frame_indices)
    if frame_indices is None:
        frame_indices = list(range(len(mesh_frames)))
    for frame_index in frame_indices:
        if frame_index < 0 or frame_index >= len(keypoints):
            raise ValueError(
                f"frame-index {frame_index} out of range for {len(keypoints)} keypoint frames"
            )
        if frame_index < 0 or frame_index >= len(mesh_frames):
            raise ValueError(
                f"frame-index {frame_index} out of range for {len(mesh_frames)} mesh frames"
            )

    keypoints_frame = keypoints[frame_indices].copy()
    keypoints_frame[:, :, :3] *= config.input.scale_to_meters
    keypoints_tensor = torch.from_numpy(keypoints_frame)
    selected_mesh_frames = [mesh_frames[frame_index] for frame_index in frame_indices]

    device = torch.device(config.fitting.device)
    mesh_targets = [
        load_mesh_target(
            mesh_frame.mesh_path,
            device=device,
            scale=config.input.scale_to_meters,
            samples=scan_surface_samples,
            seed=frame_index,
        )
        for frame_index, mesh_frame in zip(frame_indices, selected_mesh_frames)
    ]

    output_dir = args.output_dir or config.fitting.output_dir / "fitting"
    output_dir.mkdir(parents=True, exist_ok=True)

    streamer = None
    if args.aitviewer_remote or args.aitviewer_launch:
        from .visualization.aitviewer_camera_scene import (
            CameraImageOverlayConfig,
            InitialCameraConfig,
            ViewerRenderConfig,
            parse_camera_ids,
        )
        from .visualization.aitviewer_live import AitviewerRemoteFitStreamer, parse_remote_address

        host, port = parse_remote_address(args.aitviewer_remote)
        render_config = ViewerRenderConfig(
            window_type=args.aitviewer_window_type,
            shadows_enabled=args.aitviewer_shadows,
            znear=args.aitviewer_znear,
            zfar=args.aitviewer_zfar,
        )
        camera_scale = (
            args.aitviewer_camera_scale
            if args.aitviewer_camera_scale is not None
            else config.input.scale_to_meters
        )
        source_meshes = [
            (target.vertices[0].detach().cpu().numpy(), target.faces.detach().cpu().numpy())
            for target in mesh_targets
        ]
        initial_camera = None
        initial_camera_raw = args.aitviewer_initial_camera.strip()
        initial_camera_value = initial_camera_raw.lower() or "auto"
        if args.aitviewer_launch and initial_camera_value != "none":
            camera_json = args.aitviewer_camera_json or config.input.cameras
            if camera_json is None:
                if initial_camera_value != "auto":
                    raise ValueError(
                        "--aitviewer-initial-camera requires --aitviewer-camera-json "
                        "or input.cameras in the config."
                    )
            else:
                initial_camera = InitialCameraConfig(
                    camera_json=camera_json,
                    camera_id=(
                        None
                        if initial_camera_value == "auto"
                        else initial_camera_raw
                    ),
                    camera_scale=camera_scale,
                )
        camera_overlay = None
        if args.aitviewer_camera_overlay:
            camera_json = args.aitviewer_camera_json or config.input.cameras
            image_root = args.aitviewer_image_root or config.input.images
            if camera_json is None or image_root is None:
                raise ValueError(
                    "Camera overlay requires --aitviewer-camera-json/--aitviewer-image-root "
                    "or input.cameras/input.images in the config."
                )
            if not args.aitviewer_launch:
                print(
                    "Warning: --aitviewer-camera-overlay only affects viewers launched by this "
                    "process. Existing remote viewers must already contain camera overlays."
                )
            camera_overlay = CameraImageOverlayConfig(
                camera_json=camera_json,
                image_root=image_root,
                frame_ids=tuple(mesh_frames[index].frame_id for index in frame_indices),
                camera_ids=parse_camera_ids(args.aitviewer_cameras),
                max_cameras=args.aitviewer_max_cameras,
                camera_scale=camera_scale,
                billboard_distance=args.aitviewer_billboard_distance,
                billboard_alpha=args.aitviewer_billboard_alpha,
                image_extensions=tuple(
                    item.strip()
                    for item in args.aitviewer_image_extensions.split(",")
                    if item.strip()
                ),
            )
        streamer = AitviewerRemoteFitStreamer(
            host=host,
            port=port,
            timeout=args.aitviewer_timeout,
            launch=args.aitviewer_launch,
            server_log_path=args.aitviewer_log or output_dir / "aitviewer_server.log",
            source_meshes=source_meshes,
            camera_overlay=camera_overlay,
            initial_camera=initial_camera,
            render_config=render_config,
        )

    fitter = SmplFitter(config.body_model, config.fitting)
    try:
        if tracking and len(frame_indices) > 1:
            print(f"tracking enabled: fitting {len(frame_indices)} frames sequentially")
            results = fitter.fit_full_sequence(
                keypoints_tensor,
                mesh_targets=mesh_targets,
                tracking=True,
                progress_callback=streamer,
                callback_interval=args.aitviewer_update_interval,
                max_steps_per_stage=max_steps_per_stage,
                max_total_steps=max_total_steps,
                body_vertex_samples=body_vertex_samples,
            )
        else:
            results = [
                fitter.fit_full(
                    keypoints_tensor,
                    mesh_targets=mesh_targets,
                    progress_callback=streamer,
                    callback_interval=args.aitviewer_update_interval,
                    max_steps_per_stage=max_steps_per_stage,
                    max_total_steps=max_total_steps,
                    body_vertex_samples=body_vertex_samples,
                )
            ]
    except Exception:
        if streamer is not None:
            streamer.close()
        raise

    tracking_outputs = tracking and len(frame_indices) > 1
    written_outputs = _write_output_model_results(
        config,
        fitter,
        results,
        selected_mesh_frames,
        mesh_targets,
        output_dir,
        tracking_outputs,
    )
    last_loss = results[-1].loss if tracking_outputs else results[0].loss
    if streamer is not None:
        streamer.close()

    print(f"final_loss={last_loss:.6f}" if last_loss is not None else "final_loss=None")
    print(f"fit_frame_indices={frame_indices}")
    print(f"mesh_frames={[mesh_frames[index].frame_id for index in frame_indices]}")
    print(f"tracking={tracking_outputs}  shape_calibrated={fitter._betas_calibrated}")
    print(f"output_models={list(written_outputs)}")
    for model_type, model_output_dir in written_outputs.items():
        print(f"wrote_{model_type}={model_output_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    fit_command(args)


if __name__ == "__main__":
    main()
