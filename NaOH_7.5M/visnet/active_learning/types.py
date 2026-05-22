"""Structured data containers for ViSNet active learning."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass
class CandidateFrame:
    candidate_id: str
    source_round: int
    source_step: int
    species: list[str]
    positions: np.ndarray
    cell: np.ndarray
    u_frame_ev_per_angstrom: float
    u_atom_ev_per_angstrom: np.ndarray
    frame_feature: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=np.float64)
        self.cell = np.asarray(self.cell, dtype=np.float64)
        self.u_atom_ev_per_angstrom = np.asarray(self.u_atom_ev_per_angstrom, dtype=np.float64).reshape(-1)
        self.frame_feature = np.asarray(self.frame_feature, dtype=np.float32).reshape(-1)
        if self.positions.shape != (len(self.species), 3):
            raise ValueError("positions must have shape [natoms, 3] and match species length")
        if self.cell.shape != (3, 3):
            raise ValueError("cell must have shape [3, 3]")
        if self.u_atom_ev_per_angstrom.size != len(self.species):
            raise ValueError("u_atom_ev_per_angstrom must have one entry per atom")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(
            {
                "candidate_id": self.candidate_id,
                "source_round": self.source_round,
                "source_step": self.source_step,
                "species": list(self.species),
                "positions": self.positions,
                "cell": self.cell,
                "u_frame_ev_per_angstrom": self.u_frame_ev_per_angstrom,
                "u_atom_ev_per_angstrom": self.u_atom_ev_per_angstrom,
                "frame_feature": self.frame_feature,
                "metadata": self.metadata,
            }
        )


@dataclass
class LabeledFrame:
    candidate_id: str
    source_round: int
    source_step: int
    species: list[str]
    positions: np.ndarray
    cell: np.ndarray
    energy_hartree: float
    forces_hartree_per_angstrom: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=np.float64)
        self.cell = np.asarray(self.cell, dtype=np.float64)
        self.forces_hartree_per_angstrom = np.asarray(self.forces_hartree_per_angstrom, dtype=np.float64)
        if self.positions.shape != (len(self.species), 3):
            raise ValueError("positions must have shape [natoms, 3] and match species length")
        if self.forces_hartree_per_angstrom.shape != (len(self.species), 3):
            raise ValueError("forces_hartree_per_angstrom must have shape [natoms, 3]")
        if self.cell.shape != (3, 3):
            raise ValueError("cell must have shape [3, 3]")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(
            {
                "candidate_id": self.candidate_id,
                "source_round": self.source_round,
                "source_step": self.source_step,
                "species": list(self.species),
                "positions": self.positions,
                "cell": self.cell,
                "energy_hartree": self.energy_hartree,
                "forces_hartree_per_angstrom": self.forces_hartree_per_angstrom,
                "metadata": self.metadata,
            }
        )


@dataclass
class WorkflowState:
    next_round: int
    current_checkpoint: str
    current_dataset: str
    restart_state: str | None = None
    total_steps: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(
            {
                "next_round": self.next_round,
                "current_checkpoint": self.current_checkpoint,
                "current_dataset": self.current_dataset,
                "restart_state": self.restart_state,
                "total_steps": self.total_steps,
                "history": self.history,
            }
        )
