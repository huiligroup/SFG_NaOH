"""Configuration for DVR point embedding."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EmbeddingConfig:
    """Inputs and options for embedding DVR geometries into a system frame."""

    reference_xyz: str | Path
    system_xyz: str | Path
    probe_indices: list[int]
    modes: str | Path
    dvr_data: str | Path
    output_dir: str | Path | None = None
    run_name: str | None = None
    grid_unit: str = "bohr"
    mode_unit: str = "dimensionless"
    coord_unit: str = "angstrom"
    frame_index: int = 0
    masses: list[float] | None = None
    centroid_weight: str = "mass"
    write_xyz: bool = False
    prefix: str = "embedded"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.probe_indices:
            raise ValueError("probe_indices cannot be empty")
        if len(set(self.probe_indices)) != len(self.probe_indices):
            raise ValueError("probe_indices must be unique")
        if min(self.probe_indices) < 0:
            raise ValueError("probe_indices must be zero-based non-negative indices")
        if self.grid_unit.lower() not in {"bohr", "angstrom"}:
            raise ValueError(f"Unsupported grid_unit: {self.grid_unit}")
        if self.coord_unit.lower() != "angstrom":
            raise ValueError("Only Angstrom Cartesian coordinates are supported")
        if self.mode_unit.lower() not in {
            "dimensionless",
            "angstrom_per_angstrom",
            "angstrom_per_bohr",
            "bohr_per_bohr",
        }:
            raise ValueError(f"Unsupported mode_unit: {self.mode_unit}")
        if self.centroid_weight.lower() not in {"mass", "sqrt_mass", "uniform"}:
            raise ValueError(f"Unsupported centroid_weight: {self.centroid_weight}")
        if self.masses is not None and len(self.masses) != len(self.probe_indices):
            raise ValueError("masses length must match probe_indices length")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("reference_xyz", "system_xyz", "modes", "dvr_data", "output_dir"):
            if data.get(key) is not None:
                data[key] = str(data[key])
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmbeddingConfig":
        config = cls(**data)
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "EmbeddingConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def resolved_output_dir(self) -> Path:
        if self.output_dir is not None:
            return Path(self.output_dir)
        name = self.run_name or Path(self.system_xyz).stem
        return Path("NaOH_7.5M/src/lqve/data/embedded_geometries") / name

