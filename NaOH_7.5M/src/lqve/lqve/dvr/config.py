"""Configuration objects for reference DVR calculations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


DALTON_TO_ELECTRON_MASS = 1822.887427
ANGSTROM_TO_BOHR = 1.0 / 0.52917721092
CM_TO_HARTREE = 1.0 / 219474.63
EV_TO_HARTREE = 1.0 / 27.211386245988


@dataclass(frozen=True)
class ModeConfig:
    """Per-mode DVR settings.

    Coordinate ranges are stored in the user-facing unit declared by
    :class:`DVRConfig`. They are converted to Bohr inside the solver.
    """

    name: str
    lower: float
    upper: float
    sinc_points: int = 200
    podvr_points: int = 7
    mass: float = DALTON_TO_ELECTRON_MASS

    def validate(self) -> None:
        if not self.name:
            raise ValueError("Mode name cannot be empty")
        if self.upper <= self.lower:
            raise ValueError(f"Invalid range for mode {self.name}: upper <= lower")
        if self.sinc_points < 4:
            raise ValueError(f"sinc_points for {self.name} must be >= 4")
        if self.podvr_points < 1:
            raise ValueError(f"podvr_points for {self.name} must be >= 1")
        if self.podvr_points >= self.sinc_points:
            raise ValueError(
                f"podvr_points for {self.name} must be smaller than sinc_points"
            )
        if self.mass <= 0:
            raise ValueError(f"Mass for {self.name} must be positive")


@dataclass(frozen=True)
class DVRConfig:
    """Reference DVR calculation settings."""

    modes: list[ModeConfig]
    states: int = 20
    coord_unit: str = "angstrom"
    energy_unit: str = "cm-1"
    dense_threshold: int = 512
    solver_tol: float = 1.0e-10
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.modes:
            raise ValueError("At least one mode is required")
        names = [mode.name for mode in self.modes]
        if len(set(names)) != len(names):
            raise ValueError(f"Mode names must be unique: {names}")
        for mode in self.modes:
            mode.validate()
        if self.states < 1:
            raise ValueError("states must be >= 1")
        basis_size = self.basis_size
        if self.states > basis_size:
            raise ValueError(
                f"states={self.states} exceeds PODVR basis size={basis_size}"
            )
        if normalize_coord_unit(self.coord_unit) not in {"angstrom", "bohr"}:
            raise ValueError(f"Unsupported coordinate unit: {self.coord_unit}")
        if normalize_energy_unit(self.energy_unit) not in {"hartree", "cm-1", "ev"}:
            raise ValueError(f"Unsupported energy unit: {self.energy_unit}")
        if self.dense_threshold < 1:
            raise ValueError("dense_threshold must be >= 1")
        if self.solver_tol <= 0:
            raise ValueError("solver_tol must be positive")

    @property
    def dims(self) -> int:
        return len(self.modes)

    @property
    def basis_shape(self) -> tuple[int, ...]:
        return tuple(mode.podvr_points for mode in self.modes)

    @property
    def basis_size(self) -> int:
        size = 1
        for n_points in self.basis_shape:
            size *= n_points
        return size

    def ranges_bohr(self) -> list[tuple[float, float]]:
        scale = coord_to_bohr_factor(self.coord_unit)
        return [(mode.lower * scale, mode.upper * scale) for mode in self.modes]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["dims"] = self.dims
        data["basis_shape"] = list(self.basis_shape)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DVRConfig":
        raw_modes = data.get("modes")
        if not raw_modes:
            raise ValueError("Config JSON must contain a non-empty 'modes' list")
        modes = [ModeConfig(**mode) for mode in raw_modes]
        known = {
            "states",
            "coord_unit",
            "energy_unit",
            "dense_threshold",
            "solver_tol",
            "metadata",
        }
        kwargs = {key: data[key] for key in known if key in data}
        config = cls(modes=modes, **kwargs)
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "DVRConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def normalize_coord_unit(unit: str) -> str:
    normalized = unit.strip().lower()
    aliases = {"a": "angstrom", "ang": "angstrom", "angstroms": "angstrom", "bohrs": "bohr"}
    return aliases.get(normalized, normalized)


def normalize_energy_unit(unit: str) -> str:
    normalized = unit.strip().lower()
    aliases = {
        "au": "hartree",
        "a.u.": "hartree",
        "eh": "hartree",
        "ha": "hartree",
        "hartrees": "hartree",
        "cm": "cm-1",
        "cm^-1": "cm-1",
        "1/cm": "cm-1",
        "wavenumber": "cm-1",
        "wavenumbers": "cm-1",
    }
    return aliases.get(normalized, normalized)


def coord_to_bohr_factor(unit: str) -> float:
    normalized = normalize_coord_unit(unit)
    if normalized == "bohr":
        return 1.0
    if normalized == "angstrom":
        return ANGSTROM_TO_BOHR
    raise ValueError(f"Unsupported coordinate unit: {unit}")


def energy_to_hartree_factor(unit: str) -> float:
    normalized = normalize_energy_unit(unit)
    if normalized == "hartree":
        return 1.0
    if normalized == "cm-1":
        return CM_TO_HARTREE
    if normalized == "ev":
        return EV_TO_HARTREE
    raise ValueError(f"Unsupported energy unit: {unit}")

