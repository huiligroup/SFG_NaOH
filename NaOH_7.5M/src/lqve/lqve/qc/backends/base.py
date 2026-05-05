"""Base class for LQVE energy backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from lqve.qc.types import EnergyResult, GeometryBatch


class EnergyBackend(ABC):
    name = "base"

    def __init__(self, **options: Any) -> None:
        self.options = dict(options)

    @abstractmethod
    def evaluate(self, batch: GeometryBatch) -> EnergyResult:
        """Evaluate energies for all geometries in ``batch``."""
