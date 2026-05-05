"""Structured output record for LQVE analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class AnalysisResult:
    output_dir: Path
    tables: dict[str, Path] = field(default_factory=dict)
    figures: dict[str, Path] = field(default_factory=dict)
    report: Path | None = None
    statistics: dict[str, Any] = field(default_factory=dict)

    def save_metadata(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "tables": {key: str(path) for key, path in self.tables.items()},
            "figures": {key: str(path) for key, path in self.figures.items()},
            "report": None if self.report is None else str(self.report),
            "statistics": self.statistics,
        }
        with (self.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
