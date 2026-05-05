"""Shared data containers for LQVE energy evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GeometryBatch:
    """A batch of DVR-grid geometries from one AIMD frame."""

    frame_id: str
    species: list[str]
    positions: np.ndarray
    grid_points: np.ndarray
    probe_indices: np.ndarray
    cell: np.ndarray | None = None
    source_indices: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=np.float64)
        grid_points = np.asarray(self.grid_points, dtype=np.float64)
        if positions.ndim != 3 or positions.shape[-2:] != (len(self.species), 3):
            raise ValueError(
                "positions must have shape [geometries, atoms, 3] and match species length"
            )
        if grid_points.ndim != 2 or grid_points.shape[0] != positions.shape[0]:
            raise ValueError("grid_points must have shape [geometries, dims]")
        if self.cell is not None:
            cell = np.asarray(self.cell, dtype=np.float64)
            if cell.shape not in {(3,), (3, 3), (positions.shape[0], 3), (positions.shape[0], 3, 3)}:
                raise ValueError(f"Unsupported cell shape: {cell.shape}")

    @property
    def geometry_count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def atom_count(self) -> int:
        return int(self.positions.shape[1])

    def slice(self, start: int, stop: int) -> "GeometryBatch":
        cell = self.cell
        if cell is not None and np.asarray(cell).ndim == 3 and np.asarray(cell).shape[0] == self.geometry_count:
            cell = np.asarray(cell)[start:stop]
        elif (
            cell is not None
            and np.asarray(cell).ndim == 2
            and np.asarray(cell).shape != (3, 3)
            and np.asarray(cell).shape[0] == self.geometry_count
        ):
            cell = np.asarray(cell)[start:stop]
        source_indices = None if self.source_indices is None else np.asarray(self.source_indices)[start:stop]
        return GeometryBatch(
            frame_id=self.frame_id,
            species=list(self.species),
            positions=np.asarray(self.positions)[start:stop],
            grid_points=np.asarray(self.grid_points)[start:stop],
            probe_indices=np.asarray(self.probe_indices),
            cell=None if cell is None else np.asarray(cell),
            source_indices=source_indices,
            metadata=dict(self.metadata),
        )


@dataclass
class EnergyResult:
    """Structured energies for one geometry batch."""

    backend: str
    frame_id: str
    energies_hartree: np.ndarray
    grid_points: np.ndarray
    success_mask: np.ndarray
    source_indices: np.ndarray | None = None
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.energies_hartree = np.asarray(self.energies_hartree, dtype=np.float64).reshape(-1)
        self.grid_points = np.asarray(self.grid_points, dtype=np.float64)
        self.success_mask = np.asarray(self.success_mask, dtype=bool).reshape(-1)
        if self.grid_points.shape[0] != self.energies_hartree.size:
            raise ValueError("grid_points and energies_hartree length mismatch")
        if self.success_mask.size != self.energies_hartree.size:
            raise ValueError("success_mask and energies_hartree length mismatch")
        if self.source_indices is not None:
            self.source_indices = np.asarray(self.source_indices, dtype=np.int64).reshape(-1)
            if self.source_indices.size != self.energies_hartree.size:
                raise ValueError("source_indices and energies_hartree length mismatch")


def concatenate_results(results: list[EnergyResult]) -> EnergyResult:
    if not results:
        raise ValueError("No EnergyResult objects to concatenate")
    first = results[0]
    source_parts = [result.source_indices for result in results]
    source_indices = None
    if all(part is not None for part in source_parts):
        source_indices = np.concatenate([np.asarray(part) for part in source_parts])
    return EnergyResult(
        backend=first.backend,
        frame_id=first.frame_id,
        energies_hartree=np.concatenate([result.energies_hartree for result in results]),
        grid_points=np.concatenate([result.grid_points for result in results], axis=0),
        success_mask=np.concatenate([result.success_mask for result in results]),
        source_indices=source_indices,
        errors=[error for result in results for error in result.errors],
        metadata=dict(first.metadata),
    )
