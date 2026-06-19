"""Utilities for converting fitted SMPL-family outputs between model types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from mesh2smplx.core.config import BodyModelConfig, BodyModelType, FittingConfig
from .smpl_fitter import SmplFitResult, SmplFitter


MODEL_VERTEX_COUNTS: dict[str, int] = {
    "smpl": 6890,
    "smplh": 6890,
    "smplx": 10475,
}

TRANSFER_FILENAMES: dict[tuple[int, int], str] = {
    (6890, 10475): "smpl2smplx_deftrafo_setup.pkl",
    (10475, 6890): "smplx2smpl_deftrafo_setup.pkl",
}

SHARED_PARAMETER_NAMES = ("global_orient", "body_pose", "transl")
EXTRA_PARAMETER_NAMES = (
    "left_hand_pose",
    "right_hand_pose",
    "jaw_pose",
    "leye_pose",
    "reye_pose",
    "expression",
)


def infer_transfer_matrix_path(
    source_model_type: str,
    target_model_type: str,
    transfer_dir: str | Path | None,
) -> Path | None:
    """Infer the standard SMPL/SMPL-X vertex transfer matrix path."""
    source_vertices = model_vertex_count(source_model_type)
    target_vertices = model_vertex_count(target_model_type)
    if source_vertices == target_vertices:
        return None
    if transfer_dir is None:
        raise ValueError(
            f"{source_model_type}->{target_model_type} conversion requires a vertex "
            "transfer matrix. Pass --transfer-matrix or --transfer-dir."
        )
    filename = TRANSFER_FILENAMES.get((source_vertices, target_vertices))
    if filename is None:
        raise ValueError(
            f"No standard transfer matrix filename is known for "
            f"{source_model_type}->{target_model_type}."
        )
    return Path(transfer_dir) / filename


def model_vertex_count(model_type: str) -> int:
    try:
        return MODEL_VERTEX_COUNTS[model_type.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported model type: {model_type}") from exc


def load_fit_result_json(path: str | Path) -> SmplFitResult:
    """Load a saved smpl_params.json as a lightweight SmplFitResult."""
    import json

    input_path = Path(path)
    if input_path.is_dir():
        input_path = input_path / "smpl_params.json"
    data = json.loads(input_path.read_text(encoding="utf-8"))
    params = {
        key: torch.as_tensor(value, dtype=torch.float32)
        for key, value in data.get("params", {}).items()
    }
    if not params:
        raise ValueError(f"No params found in {input_path}")
    return SmplFitResult(
        params=params,
        model_type=data["model_type"].lower(),
        gender=data.get("gender", "neutral"),
    )


def result_batch_size(result: SmplFitResult) -> int:
    for value in result.params.values():
        if value.ndim == 0:
            continue
        return int(value.shape[0])
    raise ValueError("Could not infer batch size from fit result params.")


def build_target_body_config(
    source_config: BodyModelConfig,
    target_model_type: BodyModelType,
    model_path: str | Path | None = None,
    gender: str | None = None,
) -> BodyModelConfig:
    """Create a target body model config while preserving compatible knobs."""
    target_type = target_model_type.lower()
    return BodyModelConfig(
        type=target_type,  # type: ignore[arg-type]
        gender=gender or source_config.gender,
        model_path=Path(model_path) if model_path is not None else source_config.model_path,
        use_hands=source_config.use_hands and target_type in {"smplh", "smplx"},
        use_face=source_config.use_face and target_type == "smplx",
        num_betas=source_config.num_betas,
        num_expression_coeffs=source_config.num_expression_coeffs,
        num_pca_comps=source_config.num_pca_comps,
    )


def load_vertex_transfer_matrix(
    path: str | Path,
    *,
    expected_input_vertices: int | None = None,
    expected_output_vertices: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Load a sparse vertex transfer matrix as a torch CSR tensor.

    The standard SMPL/SMPL-X ``*_deftrafo_setup.pkl`` files store the matrix
    under ``mtx`` and include an extra block of columns. When the expected input
    vertex count is known, this loader keeps the first source-vertex block, which
    matches the converter behavior used by the original fitting code.
    """
    matrix = _load_sparse_matrix(Path(path))
    if expected_input_vertices is not None:
        if matrix.shape[1] == expected_input_vertices * 2:
            matrix = matrix[:, :expected_input_vertices]
        elif matrix.shape[1] != expected_input_vertices:
            raise ValueError(
                f"Transfer matrix has {matrix.shape[1]} input columns, expected "
                f"{expected_input_vertices}."
            )
    if expected_output_vertices is not None and matrix.shape[0] != expected_output_vertices:
        raise ValueError(
            f"Transfer matrix has {matrix.shape[0]} output rows, expected "
            f"{expected_output_vertices}."
        )
    return _scipy_csr_to_torch(matrix, device=device, dtype=dtype)


def convert_vertices(vertices: torch.Tensor, transfer_matrix: torch.Tensor | None) -> torch.Tensor:
    """Convert batched vertices with a sparse barycentric transfer matrix."""
    if transfer_matrix is None:
        return vertices
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError("vertices must have shape (batch_size, num_vertices, 3)")
    num_vertices_in = vertices.shape[1]
    if transfer_matrix.shape[1] != num_vertices_in:
        raise ValueError(
            f"Transfer matrix expects {transfer_matrix.shape[1]} vertices, got "
            f"{num_vertices_in}."
        )
    matrix = transfer_matrix.to(device=vertices.device, dtype=vertices.dtype)
    flat_vertices = vertices.permute(1, 0, 2).reshape(num_vertices_in, -1)
    converted = torch.sparse.mm(matrix, flat_vertices)
    num_vertices_out = transfer_matrix.shape[0]
    return converted.reshape(num_vertices_out, -1, 3).permute(1, 0, 2).contiguous()


@dataclass
class SmplModelConverter:
    """Convert a fitted SMPL-family result to another model type."""

    source_body_config: BodyModelConfig
    target_body_config: BodyModelConfig
    fitting_config: FittingConfig
    transfer_matrix_path: Path | None = None
    num_steps: int = 200
    learning_rate: float = 1.0e-2
    beta_regularizer: float = 1.0e-4
    optimize_betas: bool = True
    optimize_extra: bool = False

    def convert_result(self, result: SmplFitResult) -> SmplFitResult:
        if result.model_type.lower() != self.source_body_config.type:
            raise ValueError(
                f"Result model_type={result.model_type!r} does not match converter source "
                f"type={self.source_body_config.type!r}."
            )
        device = torch.device(self.fitting_config.device)
        source_vertices = self._source_vertices(result, device)
        transfer_matrix = self._load_transfer_matrix(device, source_vertices.dtype)
        target_vertices = convert_vertices(source_vertices, transfer_matrix)
        return self._fit_target_model(result, target_vertices)

    def _source_vertices(self, result: SmplFitResult, device: torch.device) -> torch.Tensor:
        expected_vertices = model_vertex_count(self.source_body_config.type)
        if result.vertices is not None and result.vertices.shape[1] == expected_vertices:
            return result.vertices.to(device=device, dtype=torch.float32)

        batch_size = result_batch_size(result)
        source_fitter = SmplFitter(self.source_body_config, self.fitting_config)
        source_model = source_fitter.create_body_model(batch_size=batch_size)
        source_fitter._load_params(source_model, result.params)
        with torch.no_grad():
            output = source_model(return_verts=True)
            return output.vertices.detach()

    def _load_transfer_matrix(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        source_vertices = model_vertex_count(self.source_body_config.type)
        target_vertices = model_vertex_count(self.target_body_config.type)
        if source_vertices == target_vertices:
            return None
        if self.transfer_matrix_path is None:
            raise ValueError(
                f"{self.source_body_config.type}->{self.target_body_config.type} conversion "
                "requires a transfer matrix."
            )
        return load_vertex_transfer_matrix(
            self.transfer_matrix_path,
            expected_input_vertices=source_vertices,
            expected_output_vertices=target_vertices,
            device=device,
            dtype=dtype,
        )

    def _fit_target_model(
        self,
        source_result: SmplFitResult,
        target_vertices: torch.Tensor,
    ) -> SmplFitResult:
        batch_size = target_vertices.shape[0]
        target_fitter = SmplFitter(self.target_body_config, self.fitting_config)
        target_model = target_fitter.create_body_model(batch_size=batch_size)
        target_fitter._load_params(target_model, source_result.params)
        self._align_centroid(target_model, target_vertices)

        opt_params = self._optimizable_parameters(target_model)
        last_loss = self._current_vertex_loss(target_model, target_vertices)
        if opt_params and self.num_steps > 0:
            optimizer = torch.optim.Adam(opt_params, lr=self.learning_rate, betas=(0.9, 0.999))
            for step in range(self.num_steps):
                optimizer.zero_grad(set_to_none=True)
                output = target_model(return_verts=True)
                vertex_loss = (output.vertices - target_vertices).square().mean()
                loss = vertex_loss
                if (
                    self.beta_regularizer > 0
                    and self.optimize_betas
                    and hasattr(target_model, "betas")
                ):
                    loss = loss + self.beta_regularizer * target_model.betas.square().mean()
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach().cpu())
                if step == 0 or (step + 1) % 50 == 0 or step + 1 == self.num_steps:
                    print(
                        f"conversion step {step + 1:04d}/{self.num_steps} "
                        f"loss={last_loss:.8f} vertex={float(vertex_loss.detach().cpu()):.8f}"
                    )

        with torch.no_grad():
            output = target_model(return_verts=True)
            params = {key: val.detach().cpu() for key, val in target_model.named_parameters()}
            return SmplFitResult(
                params=params,
                model_type=self.target_body_config.type,
                gender=self.target_body_config.gender,
                vertices=output.vertices.detach().cpu(),
                joints=output.joints.detach().cpu(),
                faces=target_model.faces,
                loss=last_loss,
            )

    @staticmethod
    def _align_centroid(target_model: Any, target_vertices: torch.Tensor) -> None:
        if not hasattr(target_model, "transl"):
            return
        with torch.no_grad():
            output = target_model(return_verts=True)
            delta = target_vertices.mean(dim=1) - output.vertices.mean(dim=1)
            target_model.transl.add_(delta)

    def _optimizable_parameters(self, target_model: Any) -> list[torch.Tensor]:
        names = list(SHARED_PARAMETER_NAMES)
        if self.optimize_betas:
            names.append("betas")
        if self.optimize_extra:
            names.extend(EXTRA_PARAMETER_NAMES)
        params = []
        seen = set()
        for name in names:
            if not hasattr(target_model, name):
                continue
            param = getattr(target_model, name)
            if id(param) in seen:
                continue
            params.append(param)
            seen.add(id(param))
        return params

    @staticmethod
    def _current_vertex_loss(target_model: Any, target_vertices: torch.Tensor) -> float:
        with torch.no_grad():
            output = target_model(return_verts=True)
            return float((output.vertices - target_vertices).square().mean().detach().cpu())


def _load_sparse_matrix(path: Path):
    import pickle

    import numpy as np

    try:
        if path.suffix == ".npz":
            return _load_npz_sparse_matrix(path)
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("scipy"):
            raise RuntimeError(
                "Loading SMPL vertex transfer matrices requires SciPy. Install with "
                "`python -m pip install -r requirements.txt` or `python -m pip install scipy`."
            ) from exc
        raise

    matrix = _extract_sparse_payload(payload)
    if hasattr(matrix, "tocsr"):
        return matrix.tocsr().astype(np.float32)
    return _dense_to_csr(np.asarray(matrix, dtype=np.float32))


def _load_npz_sparse_matrix(path: Path):
    import numpy as np

    try:
        import scipy.sparse as sp
    except ImportError as exc:
        raise RuntimeError(
            "Loading sparse .npz transfer matrices requires SciPy. Install with "
            "`python -m pip install -r requirements.txt` or `python -m pip install scipy`."
        ) from exc

    try:
        return sp.load_npz(path).tocsr().astype(np.float32)
    except ValueError:
        data = np.load(path)
        if {"data", "indices", "indptr", "shape"}.issubset(data.files):
            return sp.csr_matrix(
                (data["data"], data["indices"], data["indptr"]),
                shape=tuple(data["shape"]),
            ).astype(np.float32)
        raise


def _extract_sparse_payload(payload):
    if not isinstance(payload, dict):
        return payload
    for key in ("mtx", "matrix", "transfer_matrix"):
        if key in payload:
            return payload[key]
    raise ValueError("Transfer file dictionary must contain `mtx`, `matrix`, or `transfer_matrix`.")


def _dense_to_csr(array):
    try:
        import scipy.sparse as sp
    except ImportError as exc:
        raise RuntimeError(
            "Dense transfer matrix conversion requires SciPy. Install with "
            "`python -m pip install -r requirements.txt` or `python -m pip install scipy`."
        ) from exc
    if array.ndim != 2:
        raise ValueError("Transfer matrix must be 2D.")
    return sp.csr_matrix(array)


def _scipy_csr_to_torch(
    sparse_matrix,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    import numpy as np

    indptr = torch.from_numpy(sparse_matrix.indptr.astype(np.int64, copy=False))
    indices = torch.from_numpy(sparse_matrix.indices.astype(np.int64, copy=False))
    data = torch.from_numpy(sparse_matrix.data.astype(np.float32, copy=False)).to(dtype=dtype)
    if device is not None:
        indptr = indptr.to(device=device)
        indices = indices.to(device=device)
        data = data.to(device=device)
    return torch.sparse_csr_tensor(
        indptr,
        indices,
        data,
        size=sparse_matrix.shape,
    )
