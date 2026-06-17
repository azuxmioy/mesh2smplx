"""Configuration dataclasses for the registration pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

InputMode = Literal["textured_mesh"]
BodyModelType = Literal["smpl", "smplh", "smplx"]
KeypointProviderType = Literal["precomputed", "external_command", "openpose135"]


@dataclass
class InputConfig:
    mode: InputMode
    root: Path
    cameras: Path | None = None
    images: Path | None = None
    masks: Path | None = None
    meshes: Path | None = None
    textures: Path | None = None
    mesh_glob: str = "*.obj"
    texture_glob: str | None = None
    frame_id_regex: str = r".*?(\d+)$"
    scale_to_meters: float = 1.0


@dataclass
class VirtualCameraConfig:
    count: int = 24
    width: int = 1280
    height: int = 720
    focal_length: float = 1100.0
    radius: float | str = "auto"
    elevation_degrees: float = 10.0
    azimuth_offset_degrees: float = 0.0
    render_masks: bool = True
    output_dir: Path = Path("outputs/rendered")


@dataclass
class KeypointConfig:
    provider: KeypointProviderType = "precomputed"
    path: Path | None = None
    command: str | None = None
    output_dir: Path | None = None
    # openpose135 detector options.
    device: str | None = None  # None -> follow fitting.device / auto
    weights_dir: Path | None = None  # None -> $OPENPOSE135_CACHE_DIR or HF download
    hf_repo: str | None = None
    number_people_max: int = 1
    enable_hand: bool = True
    enable_face: bool = True
    confidence_threshold: float = 0.05
    render_overlays: bool = False
    overwrite: bool = False


@dataclass
class BodyModelConfig:
    type: BodyModelType = "smplx"
    gender: str = "neutral"
    model_path: Path = Path("body_models")
    use_hands: bool = True
    use_face: bool = True
    num_betas: int = 10
    num_expression_coeffs: int = 10
    num_pca_comps: int = 12
    # Shape calibration: provide pre-calibrated betas (inline list or a file with
    # a `betas` array / .npy / .json). When set, betas are held fixed (not optimized).
    betas: list[float] | None = None
    betas_path: Path | None = None


# Single place to tune the keypoint fit. ``loss_weights`` are the per-term base
# multipliers (the legacy /(1+it) schedule is applied on top, except `jaw`);
# ``schedule`` controls the staged optimisation (outer iterations + lr per stage).
DEFAULT_LOSS_WEIGHTS: dict[str, float] = {
    "pose_obj": 1.0e5,  # 3D keypoint term
    "pose_pr": 1.0,     # body_pose_prior (anti-hyperextension barriers)
    "betas": 1.0,       # shape regulariser
    "lhand": 0.1,       # left-hand pose prior (off by default in the stages)
    "rhand": 0.1,       # right-hand pose prior
    "jaw": 1.0,         # jaw regulariser
    "f_exp": 0.01,      # expression regulariser
    "limb": 1.0e4,      # limb-length term (mesh path)
    "icp": 50.0,        # scan ICP term (mesh path)
}
DEFAULT_FIT_SCHEDULE: dict[str, float] = {
    "iterations": 20,        # pose-only outer iterations
    "steps_per_iter": 30,    # Adam steps per outer iteration
    "pose_shape_iters": 5,   # pose+betas stage
    "hand_iters": 10,        # hand-refine stage (0 to skip)
    "face_iters": 5,         # face-refine stage (0 to skip)
    "pose_only_lr": 0.05,
    "pose_shape_lr": 0.02,
    "hand_lr": 0.05,
    "face_lr": 0.01,
}


@dataclass
class FittingConfig:
    device: str = "cuda"
    frames: str | None = None
    fit_mesh_refinement: bool = False
    output_dir: Path = Path("outputs/default")
    keypoint_loss_weight: float = 1_0.0
    mesh_loss_weight: float = 50.0
    limb_loss_weight: float = 10_000.0
    loss_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_LOSS_WEIGHTS))
    schedule: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_FIT_SCHEDULE))
    # Tracking: when fitting multiple frames, warm-start each frame from the
    # previous fit (and skip the keypoint-based root/translation init).
    tracking: bool = False


@dataclass
class ViewerConfig:
    enabled: bool = False


@dataclass
class PipelineConfig:
    input: InputConfig
    body_model: BodyModelConfig
    fitting: FittingConfig = field(default_factory=FittingConfig)
    keypoints: KeypointConfig = field(default_factory=KeypointConfig)
    virtual_cameras: VirtualCameraConfig | None = None
    viewer: ViewerConfig = field(default_factory=ViewerConfig)


def _as_path(value: Any) -> Path | None:
    if value is None:
        return None
    return Path(value)


def _input_config(data: dict[str, Any]) -> InputConfig:
    return InputConfig(
        mode=data["mode"],
        root=Path(data["root"]),
        cameras=_as_path(data.get("cameras")),
        images=_as_path(data.get("images")),
        masks=_as_path(data.get("masks")),
        meshes=_as_path(data.get("meshes")),
        textures=_as_path(data.get("textures")),
        mesh_glob=data.get("mesh_glob", "*.obj"),
        texture_glob=data.get("texture_glob"),
        frame_id_regex=data.get("frame_id_regex", r".*?(\d+)$"),
        scale_to_meters=float(data.get("scale_to_meters", 1.0)),
    )


def _virtual_camera_config(data: dict[str, Any] | None) -> VirtualCameraConfig | None:
    if data is None:
        return None
    if "output_dir" in data:
        data = {**data, "output_dir": Path(data["output_dir"])}
    return VirtualCameraConfig(**data)


def _keypoint_config(data: dict[str, Any] | None) -> KeypointConfig:
    data = data or {}
    return KeypointConfig(
        provider=data.get("provider", "precomputed"),
        path=_as_path(data.get("path")),
        command=data.get("command"),
        output_dir=_as_path(data.get("output_dir")),
        device=data.get("device"),
        weights_dir=_as_path(data.get("weights_dir")),
        hf_repo=data.get("hf_repo"),
        number_people_max=int(data.get("number_people_max", 1)),
        enable_hand=data.get("enable_hand", True),
        enable_face=data.get("enable_face", True),
        confidence_threshold=float(data.get("confidence_threshold", 0.05)),
        render_overlays=data.get("render_overlays", False),
        overwrite=data.get("overwrite", False),
    )


def _body_model_config(data: dict[str, Any]) -> BodyModelConfig:
    return BodyModelConfig(
        type=data.get("type", "smplx"),
        gender=data.get("gender", "neutral"),
        model_path=Path(data.get("model_path", "body_models")),
        use_hands=data.get("use_hands", True),
        use_face=data.get("use_face", True),
        num_betas=data.get("num_betas", 10),
        num_expression_coeffs=data.get("num_expression_coeffs", 10),
        num_pca_comps=data.get("num_pca_comps", 12),
        betas=data.get("betas"),
        betas_path=_as_path(data.get("betas_path")),
    )


def _fitting_config(data: dict[str, Any] | None) -> FittingConfig:
    data = data or {}
    loss_weights = {**DEFAULT_LOSS_WEIGHTS, **(data.get("loss_weights") or {})}
    schedule = {**DEFAULT_FIT_SCHEDULE, **(data.get("schedule") or {})}
    return FittingConfig(
        device=data.get("device", "cuda"),
        frames=data.get("frames"),
        fit_mesh_refinement=data.get("fit_mesh_refinement", False),
        output_dir=Path(data.get("output_dir", "outputs/default")),
        keypoint_loss_weight=float(data.get("keypoint_loss_weight", 1_000.0)),
        mesh_loss_weight=float(data.get("mesh_loss_weight", 50.0)),
        limb_loss_weight=float(data.get("limb_loss_weight", 10_000.0)),
        loss_weights={key: float(value) for key, value in loss_weights.items()},
        schedule={key: float(value) for key, value in schedule.items()},
        tracking=bool(data.get("tracking", False)),
    )


def _viewer_config(data: dict[str, Any] | None) -> ViewerConfig:
    data = data or {}
    return ViewerConfig(enabled=data.get("enabled", False))


def load_config(path: str | Path) -> PipelineConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "Loading YAML configs requires PyYAML. Install the package with project "
            "dependencies before running the CLI."
        ) from exc

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    return PipelineConfig(
        input=_input_config(data["input"]),
        virtual_cameras=_virtual_camera_config(data.get("virtual_cameras")),
        keypoints=_keypoint_config(data.get("keypoints")),
        body_model=_body_model_config(data["body_model"]),
        fitting=_fitting_config(data.get("fitting")),
        viewer=_viewer_config(data.get("viewer")),
    )
