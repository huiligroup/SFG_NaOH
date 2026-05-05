"""I/O helpers for DVR embedding."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np

BOHR_TO_ANGSTROM = 0.52917721092

_MASS_TABLE = {
    "H": 1.00784,
    "D": 2.014101778,
    "C": 12.0107,
    "N": 14.0067,
    "O": 15.999,
    "F": 18.998403163,
    "Na": 22.98976928,
    "Cl": 35.45,
}


def read_xyz(path: str | Path, frame_index: int = 0) -> tuple[list[str], np.ndarray, str]:
    if frame_index < 0:
        raise IndexError("frame_index must be non-negative")
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        current = 0
        while True:
            first = handle.readline()
            while first and not first.strip():
                first = handle.readline()
            if not first:
                raise IndexError(f"frame_index={frame_index} outside available frames 0..{current - 1}")
            try:
                n_atoms = int(first.strip())
            except ValueError as exc:
                raise ValueError(f"Invalid XYZ atom-count line in {path}: {first!r}") from exc
            comment = handle.readline()
            if not comment:
                raise ValueError(f"Missing XYZ comment line for frame {current} in {path}")
            if current == frame_index:
                species: list[str] = []
                positions = np.zeros((n_atoms, 3), dtype=np.float64)
                for atom_idx in range(n_atoms):
                    line = handle.readline()
                    if not line:
                        raise ValueError(f"Unexpected EOF in frame {current} of {path}")
                    parts = line.split()
                    if len(parts) < 4:
                        raise ValueError(f"Invalid XYZ atom line in {path}: {line!r}")
                    species.append(parts[0])
                    positions[atom_idx] = [float(parts[1]), float(parts[2]), float(parts[3])]
                return species, positions, comment.rstrip("\n")
            for _ in range(n_atoms):
                if not handle.readline():
                    raise ValueError(f"Unexpected EOF while skipping frame {current} of {path}")
            current += 1


def read_xyz_frames(path: str | Path) -> list[tuple[list[str], np.ndarray, str]]:
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    frames: list[tuple[list[str], np.ndarray, str]] = []
    idx = 0
    while idx < len(lines):
        if not lines[idx].strip():
            idx += 1
            continue
        n_atoms = int(lines[idx].strip())
        comment = lines[idx + 1] if idx + 1 < len(lines) else ""
        species: list[str] = []
        positions = np.zeros((n_atoms, 3), dtype=np.float64)
        for atom_idx in range(n_atoms):
            parts = lines[idx + 2 + atom_idx].split()
            if len(parts) < 4:
                raise ValueError(f"Invalid XYZ atom line in {path}: {lines[idx + 2 + atom_idx]!r}")
            species.append(parts[0])
            positions[atom_idx] = [float(parts[1]), float(parts[2]), float(parts[3])]
        frames.append((species, positions, comment))
        idx += n_atoms + 2
    if not frames:
        raise ValueError(f"No XYZ frames found in {path}")
    return frames


def write_xyz(path: str | Path, species: list[str], positions: np.ndarray, comment: str = "") -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(species)}\n")
        handle.write(f"{comment}\n")
        for element, xyz in zip(species, positions):
            handle.write(f"{element:2s} {xyz[0]:18.10f} {xyz[1]:18.10f} {xyz[2]:18.10f}\n")


def infer_masses(species: Iterable[str]) -> np.ndarray:
    masses = []
    for element in species:
        key = element.strip()
        if key not in _MASS_TABLE:
            raise ValueError(f"No mass available for element {element!r}; pass masses explicitly")
        masses.append(_MASS_TABLE[key])
    return np.asarray(masses, dtype=np.float64)


def read_modes(path: str | Path, n_atoms: int) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as data:
            key = "modes" if "modes" in data else list(data.files)[0]
            array = data[key]
    elif suffix in {".csv", ".txt", ".dat"}:
        delimiter = "," if suffix == ".csv" else None
        array = np.loadtxt(path, delimiter=delimiter)
    else:
        raise ValueError(f"Unsupported mode file type: {path}")
    return normalize_modes_shape(array, n_atoms)


def normalize_modes_shape(array: np.ndarray, n_atoms: int) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 1:
        if array.size != 3 * n_atoms:
            raise ValueError(f"1D mode array length must be {3 * n_atoms}, got {array.size}")
        return array.reshape(1, n_atoms, 3)
    if array.ndim == 2:
        if array.shape == (n_atoms, 3):
            return array.reshape(1, n_atoms, 3)
        if array.shape[1] == 3 * n_atoms:
            return array.reshape(array.shape[0], n_atoms, 3)
        if array.shape[0] == 3 * n_atoms:
            return array.T.reshape(array.shape[1], n_atoms, 3)
    if array.ndim == 3:
        if array.shape[1:] == (n_atoms, 3):
            return array
        if array.shape[:2] == (n_atoms, 3):
            return np.moveaxis(array, -1, 0)
    raise ValueError(f"Cannot interpret mode array shape {array.shape} for {n_atoms} atoms")


def read_dvr_grids(dvr_data: str | Path) -> list[np.ndarray]:
    dvr_data = Path(dvr_data)
    npz_path = dvr_data if dvr_data.suffix == ".npz" else dvr_data / "dvr_result.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    with np.load(npz_path) as data:
        basis_shape = tuple(int(value) for value in data["basis_shape"])
        return [np.asarray(data[f"grid_{idx}"], dtype=np.float64) for idx in range(len(basis_shape))]


def convert_grids_to_angstrom(grids: list[np.ndarray], unit: str) -> list[np.ndarray]:
    normalized = unit.lower()
    if normalized == "angstrom":
        return [np.asarray(grid, dtype=np.float64) for grid in grids]
    if normalized == "bohr":
        return [np.asarray(grid, dtype=np.float64) * BOHR_TO_ANGSTROM for grid in grids]
    raise ValueError(f"Unsupported grid unit: {unit}")


def convert_modes_to_internal(modes: np.ndarray, mode_unit: str) -> np.ndarray:
    normalized = mode_unit.lower()
    if normalized in {"dimensionless", "angstrom_per_angstrom", "bohr_per_bohr"}:
        return np.asarray(modes, dtype=np.float64)
    if normalized == "angstrom_per_bohr":
        return np.asarray(modes, dtype=np.float64) / BOHR_TO_ANGSTROM
    raise ValueError(f"Unsupported mode unit: {mode_unit}")


def write_xyz_index(path: str | Path, xyz_files: list[str], grid_points: np.ndarray) -> None:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["grid_index", "xyz_file"] + [f"q{idx}" for idx in range(grid_points.shape[1])]
        writer.writerow(header)
        for idx, file_name in enumerate(xyz_files):
            writer.writerow([idx, file_name, *grid_points[idx].tolist()])
