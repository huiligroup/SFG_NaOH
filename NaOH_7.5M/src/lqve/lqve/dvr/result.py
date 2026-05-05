"""Structured DVR calculation results."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DVRResult:
    levels_hartree: np.ndarray
    wavefunctions: np.ndarray
    grids_bohr: list[np.ndarray]
    basis_shape: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def state_count(self) -> int:
        return int(len(self.levels_hartree))

    def save(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {
            "levels_hartree": np.asarray(self.levels_hartree, dtype=np.float64),
            "wavefunctions": np.asarray(self.wavefunctions, dtype=np.float64),
            "basis_shape": np.asarray(self.basis_shape, dtype=np.int64),
            "state_count": np.asarray(self.state_count, dtype=np.int64),
        }
        for idx, grid in enumerate(self.grids_bohr):
            arrays[f"grid_{idx}"] = np.asarray(grid, dtype=np.float64)
        np.savez_compressed(output_dir / "dvr_result.npz", **arrays)
        metadata = {
            **self.metadata,
            "basis_shape": list(self.basis_shape),
            "state_count": self.state_count,
            "files": {"arrays": "dvr_result.npz", "metadata": "metadata.json"},
        }
        with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, output_dir: str | Path) -> "DVRResult":
        output_dir = Path(output_dir)
        with np.load(output_dir / "dvr_result.npz") as data:
            basis_shape = tuple(int(value) for value in data["basis_shape"])
            grids = [np.array(data[f"grid_{idx}"]) for idx in range(len(basis_shape))]
            levels = np.array(data["levels_hartree"])
            wavefunctions = np.array(data["wavefunctions"])
        metadata_path = output_dir / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        return cls(
            levels_hartree=levels,
            wavefunctions=wavefunctions,
            grids_bohr=grids,
            basis_shape=basis_shape,
            metadata=metadata,
        )

