"""Live AITviewer streaming for optimizer progress."""

from __future__ import annotations

import subprocess
import sys
import asyncio
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..fitting.smpl_fitter import SmplFitProgress
from .aitviewer_camera_scene import (
    CameraImageOverlayConfig,
    format_camera_ids,
    format_frame_ids,
)


def parse_remote_address(value: str | None, default_port: int = 8417) -> tuple[str, int]:
    if not value:
        return "localhost", default_port
    if ":" not in value:
        return value, default_port
    host, port = value.rsplit(":", 1)
    return host or "localhost", int(port)


@dataclass
class AitviewerRemoteFitStreamer:
    """Stream fitting snapshots to an AITviewer remote server.

    The viewer is intentionally optional and remote. This keeps the optimizer
    non-blocking: the fitting process sends mesh/joint updates through the
    AITviewer websocket API, while the viewer process owns the OpenGL window.
    """

    host: str = "localhost"
    port: int = 8417
    timeout: float = 10.0
    launch: bool = False
    server_log_path: Path | None = None
    source_meshes: list[tuple[np.ndarray, np.ndarray]] | None = None
    camera_overlay: CameraImageOverlayConfig | None = None

    def __post_init__(self) -> None:
        try:
            from aitviewer.remote.viewer import RemoteViewer
        except ImportError as exc:
            raise RuntimeError(
                "Live visualization requires AITviewer. Install the viewer extra "
                "or run `.venv/bin/python -m pip install aitviewer`."
            ) from exc

        self._log_handle = None
        self._process: subprocess.Popen[Any] | None = None
        if self.launch:
            if self.server_log_path is not None:
                self.server_log_path.parent.mkdir(parents=True, exist_ok=True)
                self._log_handle = self.server_log_path.open("a", encoding="utf-8")
            self._process = subprocess.Popen(
                self._launch_command(),
                stdout=self._log_handle or subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        self.viewer = RemoteViewer(
            host=self.host,
            port=self.port,
            timeout=self.timeout,
            verbose=True,
        )
        if not self.viewer.connected:
            self.close()
            raise RuntimeError(
                f"Could not connect to AITviewer remote server at {self.host}:{self.port}."
            )

        self._mesh_node = None
        self._fit_joint_node = None
        self._target_joint_node = None
        self._source_nodes = []
        self._frames: list[int] | None = None

    def _launch_command(self) -> list[str]:
        if self.camera_overlay is None:
            return [sys.executable, "-m", "aitviewer.server"]

        overlay = self.camera_overlay
        command = [
            sys.executable,
            "-m",
            "mesh2smplx.visualization.aitviewer_camera_scene",
            "--camera-json",
            str(overlay.camera_json),
            "--image-root",
            str(overlay.image_root),
            "--frame-ids",
            format_frame_ids(overlay.frame_ids),
            "--max-cameras",
            str(overlay.max_cameras),
            "--camera-scale",
            str(overlay.camera_scale),
            "--billboard-distance",
            str(overlay.billboard_distance),
            "--billboard-alpha",
            str(overlay.billboard_alpha),
            "--image-extensions",
            ",".join(overlay.image_extensions),
            "--server-port",
            str(self.port),
        ]
        camera_ids = format_camera_ids(overlay.camera_ids)
        if camera_ids is not None:
            command.extend(["--cameras", camera_ids])
        return command

    def __call__(self, progress: SmplFitProgress) -> None:
        from aitviewer.remote.renderables.meshes import RemoteMeshes
        from aitviewer.remote.renderables.spheres import RemoteSpheres

        vertices = _to_float32(progress.vertices)
        joints = _to_float32(progress.joints)
        targets = _to_float32(progress.target_joints)
        faces = np.asarray(progress.faces, dtype=np.int32)
        frames = list(range(vertices.shape[0]))

        if self._mesh_node is None:
            self._frames = frames
            self._create_source_mesh_nodes(RemoteMeshes)
            self._mesh_node = RemoteMeshes(
                self.viewer,
                vertices=vertices,
                faces=faces,
                name="live fitted SMPL-X mesh",
                color=(1.0, 0.05, 0.65, 0.82),
                draw_edges=True,
            )
            self._fit_joint_node = RemoteSpheres(
                self.viewer,
                positions=joints,
                name="live fitted SMPL-X joints",
                radius=0.012,
                color=(1.0, 0.05, 0.65, 1.0),
            )
            self._target_joint_node = RemoteSpheres(
                self.viewer,
                positions=targets,
                name="target 3D keypoints",
                radius=0.01,
                color=(0.0, 0.75, 1.0, 1.0),
            )
        else:
            self._mesh_node.update_frames(vertices=vertices, frames=frames)
            self._fit_joint_node.update_frames(positions=joints, frames=frames)
            self._target_joint_node.update_frames(positions=targets, frames=frames)

        self.viewer.set_frame(0)
        print(
            f"aitviewer step={progress.step}/{progress.total_steps} "
            f"phase={progress.phase} loss={progress.loss:.6f}"
        )

    def _create_source_mesh_nodes(self, remote_meshes_cls: Any) -> None:
        if not self.source_meshes or self._source_nodes:
            return
        for index, (vertices, faces) in enumerate(self.source_meshes):
            node = remote_meshes_cls(
                self.viewer,
                vertices=np.asarray(vertices, dtype=np.float32),
                faces=np.asarray(faces, dtype=np.int32),
                name=f"source scan mesh {index}",
                color=(0.55, 0.55, 0.55, 0.35),
                draw_edges=False,
            )
            self._source_nodes.append(node)

    def close(self) -> None:
        viewer = getattr(self, "viewer", None)
        if viewer is not None:
            try:
                if viewer.connected:
                    future = asyncio.run_coroutine_threadsafe(viewer._async_close(), viewer.loop)
                    future.result(timeout=2.0)
                    viewer.thread.join(timeout=2.0)
            except FutureTimeoutError:
                print("Warning: timed out while closing AITviewer remote connection.")
            except Exception as exc:
                print(f"Warning: failed to close AITviewer remote connection: {exc}")
        if self._log_handle is not None:
            self._log_handle.close()


def _to_float32(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)
