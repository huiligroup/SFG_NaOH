"""Deterministic test backend for LQVE energy evaluation."""

from __future__ import annotations

import numpy as np

from .base import EnergyBackend
from lqve.qc.types import EnergyResult, GeometryBatch


class MockBackend(EnergyBackend):
    name = "mock"

    def evaluate(self, batch: GeometryBatch) -> EnergyResult:
        positions = np.asarray(batch.positions, dtype=np.float64)
        weights = np.arange(1, positions.shape[1] + 1, dtype=np.float64).reshape(1, -1, 1)
        energies = (positions * weights).sum(axis=(1, 2)) * 1.0e-8
        return EnergyResult(
            backend=self.name,
            frame_id=batch.frame_id,
            energies_hartree=energies,
            grid_points=batch.grid_points,
            success_mask=np.ones(positions.shape[0], dtype=bool),
            source_indices=batch.source_indices,
            metadata={"backend": self.name, "formula": "sum(weighted_positions) * 1e-8"},
        )
