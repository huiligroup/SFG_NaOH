"""Structured LQVE shift results."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ShiftResult:
    frame_id: str
    levels_hartree: np.ndarray
    transitions_cm1: np.ndarray
    shifts_cm1: np.ndarray
    success_mask: np.ndarray
    grid_points: np.ndarray
    source_qc_dir: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return bool(np.all(self.success_mask) and np.all(np.isfinite(self.transitions_cm1)))

    def save(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_dir / "shift_result.npz",
            levels_hartree=np.asarray(self.levels_hartree, dtype=np.float64),
            transitions_cm1=np.asarray(self.transitions_cm1, dtype=np.float64),
            shifts_cm1=np.asarray(self.shifts_cm1, dtype=np.float64),
            success_mask=np.asarray(self.success_mask, dtype=bool),
            grid_points=np.asarray(self.grid_points, dtype=np.float64),
        )
        metadata = {
            **self.metadata,
            "frame_id": self.frame_id,
            "success": self.success,
            "source_qc_dir": self.source_qc_dir,
            "files": {"arrays": "shift_result.npz", "metadata": "metadata.json"},
        }
        with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, output_dir: str | Path) -> "ShiftResult":
        output_dir = Path(output_dir)
        with np.load(output_dir / "shift_result.npz") as data:
            arrays = {key: np.array(data[key]) for key in data.files}
        with (output_dir / "metadata.json").open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        return cls(
            frame_id=str(metadata["frame_id"]),
            levels_hartree=arrays["levels_hartree"],
            transitions_cm1=arrays["transitions_cm1"],
            shifts_cm1=arrays["shifts_cm1"],
            success_mask=arrays["success_mask"],
            grid_points=arrays["grid_points"],
            source_qc_dir=str(metadata.get("source_qc_dir", "")),
            metadata=metadata,
        )
