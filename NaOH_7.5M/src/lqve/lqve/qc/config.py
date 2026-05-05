"""Configuration for LQVE energy evaluation backends."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QCConfig:
    geometries: str | Path
    output_dir: str | Path = "NaOH_7.5M/src/lqve/data/qc_outputs"
    backend: str = "visnet"
    workers: int = 1
    batch_size: int = 32
    resume: bool = False
    dry_run: bool = False
    keep_workdirs: bool = False
    limit_frames: int | None = None
    limit_geometries: int | None = None
    command: str | None = None
    template: str | Path | None = None
    nproc: int = 1
    method: str = "--gfn 2"
    checkpoint: str | Path | None = None
    device: str = "cpu"
    cell: list[float] | list[list[float]] | None = None
    work_dir: str | Path | None = None
    metadata: dict[str, Any] | None = None

    def validate(self) -> None:
        if self.backend not in {"xtb", "gaussian", "cp2k", "visnet", "mock"}:
            raise ValueError(f"Unsupported backend: {self.backend}")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.nproc < 1:
            raise ValueError("nproc must be >= 1")
        if self.backend == "visnet" and self.checkpoint is None and not self.dry_run:
            raise ValueError("--checkpoint is required for --backend visnet")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("geometries", "output_dir", "template", "checkpoint", "work_dir"):
            if data.get(key) is not None:
                data[key] = str(data[key])
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QCConfig":
        config = cls(**data)
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "QCConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def resolved_output_dir(self) -> Path:
        return Path(self.output_dir)

    def resolved_work_dir(self) -> Path:
        if self.work_dir is not None:
            return Path(self.work_dir)
        return self.resolved_output_dir() / "work"
