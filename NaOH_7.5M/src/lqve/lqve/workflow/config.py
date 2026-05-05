"""Configuration objects for the end-to-end LQVE workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from lqve.dvr import DVRConfig


@dataclass(frozen=True)
class ReferenceSpec:
    reference_id: str
    pes: str | Path
    reference_xyz: str | Path
    modes: str | Path
    dvr: DVRConfig
    feature_file: str | Path | None = None
    dvr_data: str | Path | None = None
    reference_energies: str | Path | None = None
    grid_unit: str = "bohr"
    mode_unit: str = "dimensionless"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.reference_id:
            raise ValueError("reference_id cannot be empty")
        self.dvr.validate()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["dvr"] = self.dvr.to_dict()
        for key in ("pes", "reference_xyz", "modes", "feature_file", "dvr_data", "reference_energies"):
            if data.get(key) is not None:
                data[key] = str(data[key])
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReferenceSpec":
        raw = dict(data)
        raw["dvr"] = DVRConfig.from_dict(raw["dvr"])
        spec = cls(**raw)
        spec.validate()
        return spec


@dataclass(frozen=True)
class TrajectorySpec:
    traj_id: str
    path: str | Path
    probe_indices: list[int] | None = None
    mol_id: int | None = None
    reference_id: str | None = None
    feature_data: str | Path | None = None
    frame_start: int = 0
    frame_stop: int | None = None
    frame_stride: int = 1
    cell: list[float] | list[list[float]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.traj_id:
            raise ValueError("traj_id cannot be empty")
        if self.frame_start < 0:
            raise ValueError("frame_start must be non-negative")
        if self.frame_stop is not None and self.frame_stop < self.frame_start:
            raise ValueError("frame_stop must be >= frame_start")
        if self.frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")
        if self.probe_indices is None and self.mol_id is None:
            raise ValueError(f"Trajectory {self.traj_id} requires probe_indices or mol_id")
        if self.probe_indices is not None:
            if not self.probe_indices:
                raise ValueError("probe_indices cannot be empty")
            if len(set(self.probe_indices)) != len(self.probe_indices):
                raise ValueError("probe_indices must be unique")
            if min(self.probe_indices) < 0:
                raise ValueError("probe_indices must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("path", "feature_data"):
            if data.get(key) is not None:
                data[key] = str(data[key])
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrajectorySpec":
        spec = cls(**data)
        spec.validate()
        return spec


@dataclass(frozen=True)
class ReferenceSelectionWorkflowConfig:
    enabled: bool = False
    checkpoint: str | Path | None = None
    device: str = "cpu"
    top_k: int = 5

    def validate(self) -> None:
        if self.enabled and self.checkpoint is None:
            raise ValueError("reference_selection.checkpoint is required when dynamic reference is enabled")
        if self.top_k < 1:
            raise ValueError("reference_selection.top_k must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data.get("checkpoint") is not None:
            data["checkpoint"] = str(data["checkpoint"])
        return data


@dataclass(frozen=True)
class EmbeddingWorkflowConfig:
    grid_unit: str = "bohr"
    mode_unit: str = "dimensionless"
    centroid_weight: str = "mass"
    write_xyz: bool = False
    prefix: str = "embedded"


@dataclass(frozen=True)
class EnergyWorkflowConfig:
    backend: str = "visnet"
    checkpoint: str | Path | None = None
    device: str = "cpu"
    cell: list[float] | list[list[float]] | None = None
    batch_size: int = 16
    workers: int = 1
    limit_geometries: int | None = None
    command: str | None = None
    template: str | Path | None = None
    nproc: int = 1
    method: str = "--gfn 2"
    keep_workdirs: bool = False
    work_dir: str | Path | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("checkpoint", "template", "work_dir"):
            if data.get(key) is not None:
                data[key] = str(data[key])
        return data


@dataclass(frozen=True)
class ShiftWorkflowConfig:
    n_contract: int = 30
    n_transitions: int = 6
    subtract_mean: bool = True
    reference_key: str | None = None


@dataclass(frozen=True)
class WorkflowConfig:
    run_name: str
    references: list[ReferenceSpec]
    trajectories: list[TrajectorySpec]
    lqve_root: str | Path = "NaOH_7.5M/src/lqve"
    dynamic_reference: bool = False
    resume: bool = True
    overwrite: bool = False
    keep_intermediates: bool = True
    limit_frames: int | None = None
    reference_selection: ReferenceSelectionWorkflowConfig = field(default_factory=ReferenceSelectionWorkflowConfig)
    embedding: EmbeddingWorkflowConfig = field(default_factory=EmbeddingWorkflowConfig)
    energy: EnergyWorkflowConfig = field(default_factory=EnergyWorkflowConfig)
    shift: ShiftWorkflowConfig = field(default_factory=ShiftWorkflowConfig)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.run_name:
            raise ValueError("run_name cannot be empty")
        if not self.references:
            raise ValueError("At least one reference is required")
        if not self.trajectories:
            raise ValueError("At least one trajectory is required")
        ids = [reference.reference_id for reference in self.references]
        if len(set(ids)) != len(ids):
            raise ValueError(f"reference_id values must be unique: {ids}")
        for reference in self.references:
            reference.validate()
            if self.dynamic_reference and reference.feature_file is None:
                raise ValueError(f"Reference {reference.reference_id} requires feature_file when dynamic_reference=true")
        for trajectory in self.trajectories:
            trajectory.validate()
            if not self.dynamic_reference and trajectory.probe_indices is None:
                raise ValueError(
                    f"Trajectory {trajectory.traj_id} requires explicit probe_indices "
                    "when dynamic_reference is false"
                )
            if not self.dynamic_reference and trajectory.reference_id is None and len(self.references) != 1:
                raise ValueError(
                    f"Trajectory {trajectory.traj_id} needs reference_id when dynamic_reference is false "
                    "and more than one reference is configured"
                )
        if self.limit_frames is not None and self.limit_frames < 1:
            raise ValueError("limit_frames must be >= 1 when provided")
        if self.overwrite and self.resume:
            raise ValueError("overwrite and resume cannot both be true")
        self.reference_selection.validate()
        if self.dynamic_reference and not self.reference_selection.enabled:
            raise ValueError("dynamic_reference requires reference_selection.enabled=true")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["lqve_root"] = str(self.lqve_root)
        data["references"] = [reference.to_dict() for reference in self.references]
        data["trajectories"] = [trajectory.to_dict() for trajectory in self.trajectories]
        data["reference_selection"] = self.reference_selection.to_dict()
        data["energy"] = self.energy.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowConfig":
        raw = dict(data)
        raw["references"] = [ReferenceSpec.from_dict(item) for item in raw.get("references", [])]
        raw["trajectories"] = [TrajectorySpec.from_dict(item) for item in raw.get("trajectories", [])]
        raw["reference_selection"] = ReferenceSelectionWorkflowConfig(**raw.get("reference_selection", {}))
        raw["embedding"] = EmbeddingWorkflowConfig(**raw.get("embedding", {}))
        raw["energy"] = EnergyWorkflowConfig(**raw.get("energy", {}))
        raw["shift"] = ShiftWorkflowConfig(**raw.get("shift", {}))
        config = cls(**raw)
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "WorkflowConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def resolved_lqve_root(self) -> Path:
        return Path(self.lqve_root)
