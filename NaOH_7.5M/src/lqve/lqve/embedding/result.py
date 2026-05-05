"""Structured results for DVR embedding."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

from .io import write_xyz, write_xyz_index


@dataclass
class EmbeddingResult:
    embedded_probe: np.ndarray
    embedded_system: np.ndarray
    grid_points: np.ndarray
    probe_indices: np.ndarray
    reference_positions: np.ndarray
    system_positions: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    species: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(
        self,
        output_dir: str | Path,
        write_xyz_files: bool | None = None,
        prefix: str | None = None,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_dir / "embedded_geometries.npz",
            embedded_probe=self.embedded_probe,
            embedded_system=self.embedded_system,
            grid_points=self.grid_points,
            probe_indices=self.probe_indices,
            reference_positions=self.reference_positions,
            system_positions=self.system_positions,
            rotation=self.rotation,
            translation=self.translation,
        )
        metadata = dict(self.metadata)
        metadata["files"] = {
            "arrays": "embedded_geometries.npz",
            "metadata": "metadata.json",
        }
        if write_xyz_files is None:
            write_xyz_files = bool(metadata.get("write_xyz", False))
        if prefix is None:
            prefix = str(metadata.get("prefix", "embedded"))
        if write_xyz_files:
            xyz_dir = output_dir / "xyz"
            xyz_dir.mkdir(exist_ok=True)
            xyz_files = []
            for idx, positions in enumerate(self.embedded_system):
                name = f"{prefix}_{idx:06d}.xyz"
                xyz_files.append(f"xyz/{name}")
                write_xyz(
                    xyz_dir / name,
                    self.species,
                    positions,
                    comment=f"grid_index={idx} grid={self.grid_points[idx].tolist()}",
                )
            write_xyz_index(output_dir / "xyz_index.csv", xyz_files, self.grid_points)
            metadata["files"]["xyz_dir"] = "xyz"
            metadata["files"]["xyz_index"] = "xyz_index.csv"
            metadata["xyz_count"] = len(xyz_files)
        if "selected_reference" in metadata:
            with (output_dir / "selected_reference.json").open("w", encoding="utf-8") as handle:
                json.dump(metadata["selected_reference"], handle, indent=2, ensure_ascii=False)
            metadata["files"]["selected_reference"] = "selected_reference.json"
        with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, output_dir: str | Path, species: list[str] | None = None) -> "EmbeddingResult":
        output_dir = Path(output_dir)
        with np.load(output_dir / "embedded_geometries.npz") as data:
            arrays = {key: np.array(data[key]) for key in data.files}
        with (output_dir / "metadata.json").open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        return cls(
            embedded_probe=arrays["embedded_probe"],
            embedded_system=arrays["embedded_system"],
            grid_points=arrays["grid_points"],
            probe_indices=arrays["probe_indices"],
            reference_positions=arrays["reference_positions"],
            system_positions=arrays["system_positions"],
            rotation=arrays["rotation"],
            translation=arrays["translation"],
            species=species or metadata.get("species", []),
            metadata=metadata,
        )
