"""Configuration for LQVE frequency-shift calculations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ShiftConfig:
    dvr_data: str | Path | None = None
    qc_outputs: str | Path = "NaOH_7.5M/src/lqve/data/qc_outputs"
    output_dir: str | Path = "NaOH_7.5M/src/lqve/data/shifts"
    reference_energies: str | Path | None = None
    reference_key: str | None = None
    reference_mode: str = "file"
    n_contract: int = 30
    n_transitions: int = 6
    limit_frames: int | None = None
    run_name: str | None = None
    subtract_mean: bool = True
    metadata: dict[str, Any] | None = None

    def validate(self) -> None:
        if self.reference_mode not in {"file", "first-qc"}:
            raise ValueError(f"Unsupported reference_mode: {self.reference_mode}")
        if self.reference_mode == "file" and self.reference_energies is None and self.dvr_data is not None:
            raise ValueError(
                "--reference-energies is required when --reference-mode file "
                "unless per-frame selected_reference metadata supplies it"
            )
        if self.n_contract < 2:
            raise ValueError("n_contract must be >= 2")
        if self.n_transitions < 1:
            raise ValueError("n_transitions must be >= 1")
        if self.limit_frames is not None and self.limit_frames < 1:
            raise ValueError("limit_frames must be >= 1 when provided")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("dvr_data", "qc_outputs", "output_dir", "reference_energies"):
            if data.get(key) is not None:
                data[key] = str(data[key])
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ShiftConfig":
        config = cls(**data)
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "ShiftConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def resolved_output_dir(self) -> Path:
        if self.run_name:
            return Path(self.output_dir) / self.run_name
        return Path(self.output_dir)
