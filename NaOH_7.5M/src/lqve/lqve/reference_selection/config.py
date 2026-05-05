"""Configuration for ViSNet descriptor based LQVE reference selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReferenceSelectionConfig:
    reference_library: str | Path
    checkpoint: str | Path
    system_xyz: str | Path | None = None
    feature_data: str | Path | None = None
    frame_index: int = 0
    mol_id: int | None = None
    probe_indices: list[int] | None = None
    cell: str | list[float] | list[list[float]] | None = None
    device: str = "cpu"
    top_k: int = 5
    output_dir: str | Path | None = None
    distance: str = "standardized_euclidean"
    metadata: dict[str, Any] | None = None

    def validate(self) -> None:
        if self.feature_data is None and self.system_xyz is None:
            raise ValueError("reference selection requires either feature_data or system_xyz")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if self.mol_id is None and not self.probe_indices:
            raise ValueError("reference selection requires mol_id or probe_indices")
        if self.probe_indices is not None:
            if not self.probe_indices:
                raise ValueError("probe_indices cannot be empty")
            if len(set(self.probe_indices)) != len(self.probe_indices):
                raise ValueError("probe_indices must be unique")
            if min(self.probe_indices) < 0:
                raise ValueError("probe_indices must be zero-based non-negative indices")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.distance != "standardized_euclidean":
            raise ValueError(f"Unsupported reference-selection distance: {self.distance}")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("reference_library", "checkpoint", "system_xyz", "feature_data", "output_dir"):
            if data.get(key) is not None:
                data[key] = str(data[key])
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReferenceSelectionConfig":
        config = cls(**data)
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "ReferenceSelectionConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def resolved_output_dir(self) -> Path | None:
        return None if self.output_dir is None else Path(self.output_dir)
