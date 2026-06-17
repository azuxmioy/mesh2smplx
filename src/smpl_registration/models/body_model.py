"""Body model metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BodyModelSpec:
    model_path: Path
    model_type: str
    gender: str

    def validate_files_exist(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Body model path does not exist: {self.model_path}. "
                "Download SMPL/SMPL-X models separately under their license."
            )

