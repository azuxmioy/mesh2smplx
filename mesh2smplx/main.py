"""Command line entry point for mesh-to-SMPL-X fitting."""

from __future__ import annotations

import argparse
from pathlib import Path

from .core.config import load_config
from .core.frame_selection import parse_frame_range
from .core.pipeline import Pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mesh2smplx")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the registration pipeline.")
    run_parser.add_argument("--config", required=True, type=Path, help="Path to a YAML config.")
    run_parser.add_argument(
        "--frames",
        default=None,
        help="Optional frame override, for example '0,20,40' or '190-450'.",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load config and print the planned stages without running heavy work.",
    )

    view_parser = subparsers.add_parser("view", help="Open outputs in AITviewer.")
    view_parser.add_argument("--config", required=True, type=Path, help="Path to a YAML config.")
    view_parser.add_argument("--outputs", required=True, type=Path, help="Pipeline output directory.")

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect mesh sequence discovery without rendering or fitting.",
    )
    inspect_parser.add_argument("--config", required=True, type=Path, help="Path to a YAML config.")
    inspect_parser.add_argument(
        "--frames",
        default=None,
        help="Optional frame override, for example '81,101' or '81-221'.",
    )

    full_parser = subparsers.add_parser(
        "fit-full",
        help="Full end-to-end SMPL-X fit: staged keypoint + mesh (ICP) schedule. "
        "Loss weights come from fitting.loss_weights in the config.",
    )
    full_parser.add_argument("--config", required=True, type=Path, help="Path to a YAML config.")
    full_parser.add_argument("--keypoints3d", required=True, type=Path, help="Path to keypoints .npy.")
    full_parser.add_argument(
        "--frame-indices",
        default="0",
        help="Index list/range into the keypoint and mesh sequence, for example '0,2-4'.",
    )
    full_parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory.")
    full_parser.add_argument(
        "--scan-surface-samples",
        "--mesh-samples",
        dest="scan_surface_samples",
        type=int,
        default=5000,
        help="Target scan surface samples for the approximate mesh loss.",
    )
    full_parser.add_argument(
        "--body-vertex-samples",
        "--mesh-vertex-samples",
        dest="body_vertex_samples",
        type=int,
        default=5000,
        help="SMPL/SMPL-X body vertices sampled for the approximate mesh loss.",
    )
    full_parser.add_argument(
        "--max-steps-per-stage",
        type=int,
        default=None,
        help="Optional cap for each schedule stage, useful for smoke tests.",
    )
    full_parser.add_argument(
        "--max-total-steps",
        type=int,
        default=None,
        help="Optional cap across the whole schedule. NOTE: this is spent in order, so a "
        "small value never reaches ICP/hands/face — use --max-steps-per-stage for that.",
    )
    full_parser.add_argument(
        "--tracking",
        action="store_true",
        help="Fit frames sequentially, warm-starting each frame from the previous one "
        "(skips the root/translation init after frame 0). Also enabled via fitting.tracking.",
    )
    full_parser.add_argument(
        "--betas",
        type=Path,
        default=None,
        help="Shape calibration: path to betas (.npy/.npz with `betas`/.json). Overrides the "
        "config; betas are held fixed (not optimized).",
    )
    full_parser.add_argument(
        "--aitviewer-remote",
        default=None,
        help="Optional AITviewer remote server as HOST:PORT, for example localhost:8417.",
    )
    full_parser.add_argument("--aitviewer-launch", action="store_true", help="Launch a local AITviewer server.")
    full_parser.add_argument("--aitviewer-update-interval", type=int, default=25)
    full_parser.add_argument("--aitviewer-timeout", type=float, default=10.0)
    full_parser.add_argument("--aitviewer-log", type=Path, default=None)
    full_parser.add_argument(
        "--aitviewer-camera-overlay",
        action="store_true",
        help="When launching AITviewer, add calibrated cameras and image billboards.",
    )
    full_parser.add_argument(
        "--aitviewer-camera-json",
        type=Path,
        default=None,
        help="Override camera calibration JSON. Defaults to input.cameras.",
    )
    full_parser.add_argument(
        "--aitviewer-image-root",
        type=Path,
        default=None,
        help="Override camera image root. Defaults to input.images.",
    )
    full_parser.add_argument(
        "--aitviewer-cameras",
        default=None,
        help="Comma-separated camera ids to show. Defaults to the first --aitviewer-max-cameras ids.",
    )
    full_parser.add_argument("--aitviewer-max-cameras", type=int, default=4)
    full_parser.add_argument("--aitviewer-camera-scale", type=float, default=None)
    full_parser.add_argument("--aitviewer-billboard-distance", type=float, default=2.0)
    full_parser.add_argument("--aitviewer-billboard-alpha", type=float, default=0.55)
    full_parser.add_argument("--aitviewer-image-extensions", default=".png,.jpg,.jpeg")

    return parser


def run_command(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.frames is not None:
        config.fitting.frames = args.frames

    pipeline = Pipeline(config)
    if args.dry_run:
        for stage in pipeline.plan():
            print(stage)
        return

    pipeline.run()


def view_command(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    from .visualization.aitviewer import open_viewer

    open_viewer(config=config, output_dir=args.outputs)


def inspect_command(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.frames is not None:
        config.fitting.frames = args.frames

    frames = parse_frame_range(config.fitting.frames)
    bundle = Pipeline(config).build_source().load(frames=frames)

    print(f"input: {config.input.mode}")
    print(f"frames: {bundle.frame_ids}")
    print(f"num_frames: {len(bundle.frames)}")
    print(f"num_virtual_cameras: {len(bundle.cameras)}")

    missing_textures = [frame.frame_id for frame in bundle.frames if frame.texture_path is None]
    print(f"missing_textures: {missing_textures}")

    for frame in bundle.frames:
        texture_name = frame.texture_path.name if frame.texture_path is not None else "None"
        print(f"{frame.frame_id}: {frame.mesh_path.name} | {texture_name}")


def fit_full_command(args: argparse.Namespace) -> None:
    import numpy as np
    import torch

    from .core.data.mesh_sequence import discover_mesh_sequence
    from .fitting import SmplFitter
    from .fitting.mesh_losses import load_mesh_target

    config = load_config(args.config)
    if args.betas is not None:
        config.body_model.betas_path = args.betas
    tracking = args.tracking or config.fitting.tracking

    keypoints = np.load(args.keypoints3d)
    frame_indices = parse_frame_range(args.frame_indices)
    mesh_frames = discover_mesh_sequence(config.input)
    for frame_index in frame_indices:
        if frame_index < 0 or frame_index >= len(keypoints):
            raise ValueError(f"frame-index {frame_index} out of range for {len(keypoints)} keypoint frames")
        if frame_index < 0 or frame_index >= len(mesh_frames):
            raise ValueError(f"frame-index {frame_index} out of range for {len(mesh_frames)} mesh frames")

    keypoints_frame = keypoints[frame_indices].copy()
    keypoints_frame[:, :, :3] *= config.input.scale_to_meters
    keypoints_tensor = torch.from_numpy(keypoints_frame)

    device = torch.device(config.fitting.device)
    mesh_targets = [
        load_mesh_target(
            mesh_frames[frame_index].mesh_path,
            device=device,
            scale=config.input.scale_to_meters,
            samples=args.scan_surface_samples,
            seed=frame_index,
        )
        for frame_index in frame_indices
    ]

    output_dir = args.output_dir or config.fitting.output_dir / "fit_full"
    output_dir.mkdir(parents=True, exist_ok=True)

    streamer = None
    if args.aitviewer_remote or args.aitviewer_launch:
        from .visualization.aitviewer_camera_scene import (
            CameraImageOverlayConfig,
            parse_camera_ids,
        )
        from .visualization.aitviewer_live import AitviewerRemoteFitStreamer, parse_remote_address

        host, port = parse_remote_address(args.aitviewer_remote)
        source_meshes = [
            (target.vertices[0].detach().cpu().numpy(), target.faces.detach().cpu().numpy())
            for target in mesh_targets
        ]
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
                camera_scale=(
                    args.aitviewer_camera_scale
                    if args.aitviewer_camera_scale is not None
                    else config.input.scale_to_meters
                ),
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
                max_steps_per_stage=args.max_steps_per_stage,
                max_total_steps=args.max_total_steps,
                body_vertex_samples=args.body_vertex_samples,
            )
        else:
            results = [
                fitter.fit_full(
                    keypoints_tensor,
                    mesh_targets=mesh_targets,
                    progress_callback=streamer,
                    callback_interval=args.aitviewer_update_interval,
                    max_steps_per_stage=args.max_steps_per_stage,
                    max_total_steps=args.max_total_steps,
                    body_vertex_samples=args.body_vertex_samples,
                )
            ]
    except Exception:
        if streamer is not None:
            streamer.close()
        raise

    if tracking and len(frame_indices) > 1:
        # One result per frame (each batch size 1).
        for result, frame_index in zip(results, frame_indices):
            fitter.save_result(result, output_dir / f"frame_index_{frame_index:06d}")
            fitter.save_mesh_obj(
                result, output_dir / f"frame_index_{frame_index:06d}.obj", batch_index=0
            )
        last_loss = results[-1].loss
    else:
        # One batched result covering all frames.
        result = results[0]
        fitter.save_result(result, output_dir)
        for batch_index, frame_index in enumerate(frame_indices):
            fitter.save_mesh_obj(
                result,
                output_dir / f"frame_index_{frame_index:06d}.obj",
                batch_index=batch_index,
            )
        last_loss = result.loss
    if streamer is not None:
        streamer.close()

    print(f"final_loss={last_loss:.6f}" if last_loss is not None else "final_loss=None")
    print(f"fit_frame_indices={frame_indices}")
    print(f"mesh_frames={[mesh_frames[index].frame_id for index in frame_indices]}")
    print(f"tracking={tracking and len(frame_indices) > 1}  shape_calibrated={fitter._betas_calibrated}")
    print(f"wrote={output_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        run_command(args)
    elif args.command == "view":
        view_command(args)
    elif args.command == "inspect":
        inspect_command(args)
    elif args.command == "fit-full":
        fit_full_command(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
