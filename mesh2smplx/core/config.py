"""Configuration dataclasses for the registration pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

InputMode = Literal["textured_mesh"]
BodyModelType = Literal["smpl", "smplh", "smplx"]
KeypointProviderType = Literal["auto", "precomputed", "external_command", "openpose135"]
RenderingMode = Literal["auto", "real", "virtual"]

MODEL_TYPE_ALIASES: dict[str, BodyModelType] = {
    "smpl": "smpl",
    "smplh": "smplh",
    "smpl-h": "smplh",
    "smplx": "smplx",
    "smpl-x": "smplx",
}


@dataclass
class InputConfig:
    mode: InputMode = "textured_mesh"
    root: Path = Path("data")
    cameras: Path | None = None
    images: Path | None = None
    masks: Path | None = None
    meshes: Path | None = Path("data/meshes")
    textures: Path | None = None
    keypoints_2d: Path | None = Path("data/keypoints_2d")
    keypoints_3d: Path | None = Path("data/keypoints_3d.npy")
    body_shape: Path | None = None
    image_glob: str = "*.png"
    mesh_glob: str = "**/*"
    texture_glob: str | None = None
    frame_id_regex: str = r".*?(\d+)$"
    scale_to_meters: float = 1.0


@dataclass
class RenderingConfig:
    mode: RenderingMode = "auto"
    count: int = 24
    width: int = 1280
    height: int = 720
    focal_length: float = 1100.0
    radius: float | str = "auto"
    elevation_degrees: float = 10.0
    azimuth_offset_degrees: float = 0.0
    render_masks: bool = True
    background_color: tuple[float, float, float] = (0.0, 1.0, 0.0)
    output_dir: Path = Path("data/rendered")


VirtualCameraConfig = RenderingConfig


@dataclass
class KeypointConfig:
    provider: KeypointProviderType = "auto"
    path: Path | None = None
    command: str | None = None
    output_dir: Path | None = None
    # openpose135 detector options.
    device: str | None = None  # None -> follow fitting.device / auto
    weights_dir: Path | None = None  # None -> checkpoints/openpose135
    hf_repo: str | None = None
    number_people_max: int = 1
    enable_hand: bool = True
    enable_face: bool = True
    confidence_threshold: float = 0.05
    render_overlays: bool = False
    overwrite: bool = False
    crop_to_mask: bool = True
    crop_padding: float = 0.15
    crop_padding_pixels: int | None = 30
    crop_aspect_height: int | None = 1200
    crop_aspect_width: int | None = 900
    max_input_size: int | None = 1280


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


# Single place to tune the staged mesh-aware fit. ``loss_weights`` are per-term
# base multipliers; the legacy /(1+it) schedule is applied on top, except `jaw`.
DEFAULT_LOSS_WEIGHTS: dict[str, float] = {
    "pose_obj": 1.0e5,  # 3D keypoint term
    "pose_pr": 1.0,     # body_pose_prior (anti-hyperextension barriers)
    "spine_pose": 25.0, # extra L2 on spine1/spine2/spine3 to reduce torso bending
    "neck_pose": 25.0,  # extra L2 on neck/head to reduce face-keypoint overfitting
    "betas": 1.0,       # shape regulariser
    "lhand": 0.1,       # left-hand pose prior for hand stages
    "rhand": 0.1,       # right-hand pose prior for hand stages
    "jaw": 1.0,         # jaw regulariser
    "f_exp": 0.01,      # expression regulariser
    "limb": 1.0e4,      # limb-length term (mesh path)
    "icp": 50.0,        # scan ICP term (mesh path)
}


@dataclass
class FittingConfig:
    device: str = "cuda"
    frames: str | None = None
    output_dir: Path = Path("data")
    scan_surface_samples: int = 2000
    body_vertex_samples: int = 2000
    max_steps_per_stage: int | None = None
    max_total_steps: int | None = None
    loss_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_LOSS_WEIGHTS))
    # Tracking: when fitting multiple frames, warm-start each frame from the
    # previous fit (and skip the keypoint-based root/translation init).
    tracking: bool = True


@dataclass
class ConversionConfig:
    # Desired exported model types. If a requested type differs from
    # body_model.type, the fitting entry point automatically runs the converter.
    output_types: list[BodyModelType] = field(default_factory=lambda: ["smplx"])
    model_path: Path | None = None
    gender: str | None = None
    transfer_matrix: Path | None = None
    transfer_dir: Path | None = None
    output_dir: Path | None = None
    num_steps: int = 200
    learning_rate: float = 1.0e-2
    beta_regularizer: float = 1.0e-4
    optimize_betas: bool = True


@dataclass
class ViewerConfig:
    enabled: bool = False
    remote: str | None = None
    update_interval: int = 25
    timeout: float = 10.0
    log_path: Path | None = None
    window_type: str | None = None
    camera_overlay: bool = False
    camera_json: Path | None = None
    image_root: Path | None = None
    cameras: str | None = None
    max_cameras: int = 4
    calibration_scale: float | None = None
    initial_camera: str = "auto"
    billboard_distance: float = 2.0
    billboard_alpha: float = 0.55
    image_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg")
    shadows: bool = False
    znear: float = 0.05
    zfar: float = 50.0


@dataclass
class PipelineConfig:
    input: InputConfig
    body_model: BodyModelConfig
    fitting: FittingConfig = field(default_factory=FittingConfig)
    keypoints: KeypointConfig = field(default_factory=KeypointConfig)
    conversion: ConversionConfig = field(default_factory=ConversionConfig)
    rendering: RenderingConfig | None = None
    viewer: ViewerConfig = field(default_factory=ViewerConfig)

    @property
    def virtual_cameras(self) -> RenderingConfig | None:
        return self.rendering


def _as_path(value: Any) -> Path | None:
    if value is None:
        return None
    return Path(value)


def _optional_child_path(root: Path, child: str) -> Path | None:
    path = root / child
    return path if path.exists() else None


def _auto_camera_path(root: Path) -> Path | None:
    calibration_dir = root / "calibration"
    if not calibration_dir.exists():
        return None
    candidates = [
        calibration_dir / "cameras.json",
        calibration_dir / "rgb_cameras.json",
        calibration_dir / "calibration.json",
        calibration_dir / "camera.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    json_files = sorted(calibration_dir.glob("*.json"))
    return json_files[0] if len(json_files) == 1 else None


def canonical_body_model_type(value: str) -> BodyModelType:
    normalized = str(value).strip().lower().replace("_", "-")
    try:
        return MODEL_TYPE_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported body model type: {value}") from exc


def parse_body_model_types(value: Any) -> list[BodyModelType]:
    if value is None:
        return []
    values = value.split(",") if isinstance(value, str) else list(value)
    output_types = []
    seen = set()
    for item in values:
        if item is None or str(item).strip() == "":
            continue
        model_type = canonical_body_model_type(str(item))
        if model_type in seen:
            continue
        output_types.append(model_type)
        seen.add(model_type)
    return output_types


def _input_config(data: dict[str, Any] | None) -> InputConfig:
    data = data or {}
    root = Path(data.get("root", "data"))
    cameras = data.get("cameras") or data.get("calibration")
    if cameras is None:
        cameras = _auto_camera_path(root)
    meshes = data.get("meshes") or root / "meshes"
    images = data["images"] if "images" in data else _optional_child_path(root, "images")
    textures = data["textures"] if "textures" in data else _optional_child_path(root, "textures")
    keypoints_2d = data.get("keypoints_2d") or root / "keypoints_2d"
    keypoints_3d = data.get("keypoints_3d") or root / "keypoints_3d.npy"
    body_shape = data.get("body_shape") or _optional_child_path(root, "body_shape.npy")
    return InputConfig(
        mode=data.get("mode", "textured_mesh"),
        root=root,
        cameras=_as_path(cameras),
        images=_as_path(images),
        masks=_as_path(data.get("masks")),
        meshes=_as_path(meshes),
        textures=_as_path(textures),
        keypoints_2d=_as_path(keypoints_2d),
        keypoints_3d=_as_path(keypoints_3d),
        body_shape=_as_path(body_shape),
        image_glob=data.get("image_glob", "*.png"),
        mesh_glob=data.get("mesh_glob", "**/*"),
        texture_glob=data.get("texture_glob"),
        frame_id_regex=data.get("frame_id_regex", r".*?(\d+)$"),
        scale_to_meters=float(data.get("scale_to_meters", 1.0)),
    )


def _rendering_config(
    data: dict[str, Any] | None,
    input_config: InputConfig,
) -> RenderingConfig | None:
    data = data or {}
    if "rendering" in data and isinstance(data["rendering"], dict):
        data = data["rendering"]
    mode = str(data.get("mode", "auto")).lower().strip()
    mode_aliases = {
        "calibrated": "real",
        "calibration": "real",
        "camera": "real",
        "cameras": "real",
        "heuristic": "virtual",
        "semi_sphere": "virtual",
        "semi-sphere": "virtual",
    }
    mode = mode_aliases.get(mode, mode)
    if mode not in {"auto", "real", "virtual"}:
        raise ValueError("rendering.mode must be one of: auto, real, virtual.")
    output_dir = data.get("output_dir") or input_config.root / "rendered"
    data = {
        **data,
        "mode": mode,
        "output_dir": Path(output_dir),
        "background_color": _parse_rgb_color(data.get("background_color", "green")),
    }
    return RenderingConfig(**data)


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "Loading YAML configs requires PyYAML. Install the package with project "
            "dependencies before running the CLI."
        ) from exc

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping.")
    return data


def _merged_rendering_data(data: dict[str, Any], config_path: Path) -> dict[str, Any] | None:
    sidecar = config_path.parent / "rendering.yaml"
    sidecar_data = _load_yaml_mapping(sidecar) if sidecar.exists() else {}
    if "rendering" in sidecar_data and isinstance(sidecar_data["rendering"], dict):
        sidecar_data = sidecar_data["rendering"]
    legacy_data = data.get("virtual_cameras") or {}
    rendering_data = data.get("rendering") or {}
    if not isinstance(legacy_data, dict) or not isinstance(rendering_data, dict):
        raise ValueError("rendering and virtual_cameras configs must be YAML mappings.")
    return {**sidecar_data, **legacy_data, **rendering_data}


def _parse_rgb_color(value: Any) -> tuple[float, float, float]:
    named = {
        "black": (0.0, 0.0, 0.0),
        "green": (0.0, 1.0, 0.0),
        "white": (1.0, 1.0, 1.0),
        "gray": (0.5, 0.5, 0.5),
        "grey": (0.5, 0.5, 0.5),
    }
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in named:
            return named[normalized]
        parts = [part.strip() for part in normalized.split(",") if part.strip()]
        if len(parts) == 3:
            value = [float(part) for part in parts]
        else:
            raise ValueError(
                "background_color must be a color name or RGB triplet, "
                f"got {value!r}."
            )
    if isinstance(value, (list, tuple)) and len(value) == 3:
        color = tuple(float(channel) for channel in value)
        if any(channel > 1.0 for channel in color):
            color = tuple(channel / 255.0 for channel in color)
        if any(channel < 0.0 or channel > 1.0 for channel in color):
            raise ValueError(
                "background_color channels must be in [0, 1] or [0, 255], "
                f"got {value!r}."
            )
        return color
    raise ValueError(f"background_color must be a color name or RGB triplet, got {value!r}.")


def _keypoint_config(data: dict[str, Any] | None, input_config: InputConfig) -> KeypointConfig:
    data = data or {}
    crop_padding_pixels = data.get("crop_padding_pixels", 30)
    crop_aspect_height = data.get("crop_aspect_height", 1200)
    crop_aspect_width = data.get("crop_aspect_width", 900)
    max_input_size = data.get("max_input_size", 1280)
    return KeypointConfig(
        provider=data.get("provider", "auto"),
        path=_as_path(data.get("path")) or input_config.keypoints_2d,
        command=data.get("command"),
        output_dir=_as_path(data.get("output_dir")) or input_config.keypoints_2d,
        device=data.get("device"),
        weights_dir=_as_path(data.get("weights_dir")),
        hf_repo=data.get("hf_repo"),
        number_people_max=int(data.get("number_people_max", 1)),
        enable_hand=data.get("enable_hand", True),
        enable_face=data.get("enable_face", True),
        confidence_threshold=float(data.get("confidence_threshold", 0.05)),
        render_overlays=data.get("render_overlays", False),
        overwrite=data.get("overwrite", False),
        crop_to_mask=bool(data.get("crop_to_mask", True)),
        crop_padding=float(data.get("crop_padding", 0.15)),
        crop_padding_pixels=(
            None if crop_padding_pixels is None else int(crop_padding_pixels)
        ),
        crop_aspect_height=(
            None if crop_aspect_height is None else int(crop_aspect_height)
        ),
        crop_aspect_width=(
            None if crop_aspect_width is None else int(crop_aspect_width)
        ),
        max_input_size=(
            None if max_input_size is None else int(max_input_size)
        ),
    )


def _body_model_config(data: dict[str, Any]) -> BodyModelConfig:
    return BodyModelConfig(
        type=canonical_body_model_type(data.get("type", "smplx")),
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


def _fitting_config(data: dict[str, Any] | None, input_config: InputConfig) -> FittingConfig:
    data = data or {}
    loss_weights = {**DEFAULT_LOSS_WEIGHTS, **(data.get("loss_weights") or {})}
    return FittingConfig(
        device=data.get("device", "cuda"),
        frames=data.get("frames"),
        output_dir=Path(data.get("output_dir") or input_config.root),
        scan_surface_samples=int(data.get("scan_surface_samples", 2000)),
        body_vertex_samples=int(data.get("body_vertex_samples", 2000)),
        max_steps_per_stage=(
            None if data.get("max_steps_per_stage") is None else int(data["max_steps_per_stage"])
        ),
        max_total_steps=(
            None if data.get("max_total_steps") is None else int(data["max_total_steps"])
        ),
        loss_weights={key: float(value) for key, value in loss_weights.items()},
        tracking=bool(data.get("tracking", True)),
    )


def _conversion_config(data: dict[str, Any] | None) -> ConversionConfig:
    data = data or {}
    output_types = parse_body_model_types(data.get("output_types"))
    if not output_types:
        # Backward-compatible alias from the first draft of the conversion config.
        output_types = parse_body_model_types(data.get("target_type")) or ["smplx"]
    return ConversionConfig(
        output_types=output_types,
        model_path=_as_path(data.get("model_path")),
        gender=data.get("gender"),
        transfer_matrix=_as_path(data.get("transfer_matrix")),
        transfer_dir=_as_path(data.get("transfer_dir")),
        output_dir=_as_path(data.get("output_dir")),
        num_steps=int(data.get("num_steps", 200)),
        learning_rate=float(data.get("learning_rate", 1.0e-2)),
        beta_regularizer=float(data.get("beta_regularizer", 1.0e-4)),
        optimize_betas=bool(data.get("optimize_betas", True)),
    )


def _viewer_config(data: dict[str, Any] | None) -> ViewerConfig:
    data = data or {}
    if "viewer" in data and isinstance(data["viewer"], dict):
        data = data["viewer"]
    calibration_scale = data.get("calibration_scale", data.get("camera_scale"))
    image_extensions = data.get("image_extensions", (".png", ".jpg", ".jpeg"))
    if image_extensions is None:
        image_extensions = (".png", ".jpg", ".jpeg")
    if isinstance(image_extensions, str):
        image_extensions = tuple(
            item.strip() for item in image_extensions.split(",") if item.strip()
        )
    else:
        image_extensions = tuple(str(item) for item in image_extensions)
    return ViewerConfig(
        enabled=bool(data.get("enabled", False)),
        remote=data.get("remote"),
        update_interval=int(data.get("update_interval", 25)),
        timeout=float(data.get("timeout", 10.0)),
        log_path=_as_path(data.get("log_path", data.get("log"))),
        window_type=data.get("window_type"),
        camera_overlay=bool(data.get("camera_overlay", data.get("overlay", False))),
        camera_json=_as_path(data.get("camera_json")),
        image_root=_as_path(data.get("image_root")),
        cameras=data.get("cameras"),
        max_cameras=int(data.get("max_cameras", 4)),
        calibration_scale=(
            None if calibration_scale is None else float(calibration_scale)
        ),
        initial_camera=str(data.get("initial_camera", "auto")),
        billboard_distance=float(data.get("billboard_distance", 2.0)),
        billboard_alpha=float(data.get("billboard_alpha", 0.55)),
        image_extensions=image_extensions,
        shadows=bool(data.get("shadows", False)),
        znear=float(data.get("znear", 0.05)),
        zfar=float(data.get("zfar", 50.0)),
    )


def load_viewer_config(path: str | Path) -> ViewerConfig:
    data = _load_yaml_mapping(Path(path))
    return _viewer_config(data)


def load_config(path: str | Path) -> PipelineConfig:
    path = Path(path)
    data = _load_yaml_mapping(path)

    input_config = _input_config(data.get("input"))
    body_model_config = _body_model_config(data["body_model"])
    if (
        input_config.body_shape is not None
        and body_model_config.betas is None
        and body_model_config.betas_path is None
    ):
        body_model_config.betas_path = input_config.body_shape
    return PipelineConfig(
        input=input_config,
        rendering=_rendering_config(_merged_rendering_data(data, path), input_config),
        keypoints=_keypoint_config(data.get("keypoints"), input_config),
        body_model=body_model_config,
        fitting=_fitting_config(data.get("fitting"), input_config),
        conversion=_conversion_config(data.get("conversion")),
        viewer=_viewer_config(data.get("viewer")),
    )
