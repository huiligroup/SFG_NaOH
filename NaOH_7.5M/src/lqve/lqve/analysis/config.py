"""Configuration for LQVE result analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AnalysisConfig:
    shifts: str | Path
    output_dir: str | Path = "NaOH_7.5M/src/lqve/results"
    transitions: list[int] | None = None
    frame_axis: str = "index"
    plot_format: str = "png"
    report_title: str = "LQVE Analysis Report"
    run_name: str | None = None
    dvr_data: str | Path | None = None
    include_dvr_diagnostics: bool = False
    metadata: dict[str, Any] | None = None

    def validate(self) -> None:
        if self.frame_axis not in {"index", "frame_id"}:
            raise ValueError(f"Unsupported frame_axis: {self.frame_axis}")
        if self.plot_format.lower() not in {"png", "pdf", "svg"}:
            raise ValueError(f"Unsupported plot_format: {self.plot_format}")
        if self.transitions is not None:
            if not self.transitions:
                raise ValueError("transitions cannot be empty when provided")
            if min(self.transitions) < 1:
                raise ValueError("transitions are one-based and must be >= 1")
        if self.include_dvr_diagnostics and self.dvr_data is None:
            raise ValueError("dvr_data is required when include_dvr_diagnostics is true")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("shifts", "output_dir", "dvr_data"):
            if data.get(key) is not None:
                data[key] = str(data[key])
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisConfig":
        config = cls(**data)
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "AnalysisConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def resolved_output_dir(self) -> Path:
        if self.run_name:
            return Path(self.output_dir) / self.run_name
        return Path(self.output_dir)
