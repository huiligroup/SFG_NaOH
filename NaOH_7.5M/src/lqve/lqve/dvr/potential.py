"""Potential energy surface readers and interpolators for DVR calculations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RegularGridInterpolator, interp1d

from .config import coord_to_bohr_factor, energy_to_hartree_factor


@dataclass(frozen=True)
class PotentialSurface:
    path: Path
    dims: int
    coord_unit: str
    energy_unit: str
    coord_columns: list[str]
    energy_column: str
    coordinates_bohr: np.ndarray
    energies_hartree: np.ndarray
    is_regular_grid: bool
    metadata: dict[str, Any]
    _interpolator: Any
    _nearest_interpolator: Any = None

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        dims: int,
        coord_unit: str,
        energy_unit: str,
    ) -> "PotentialSurface":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        frame = _read_table(path)
        if frame.shape[1] < dims + 1:
            raise ValueError(
                f"PES table {path} has {frame.shape[1]} columns; need at least {dims + 1}"
            )

        coord_columns = list(frame.columns[:dims])
        energy_column = str(frame.columns[dims])
        coords = frame.iloc[:, :dims].to_numpy(dtype=np.float64)
        energies = frame.iloc[:, dims].to_numpy(dtype=np.float64)
        coords_bohr = coords * coord_to_bohr_factor(coord_unit)
        energies_hartree = energies * energy_to_hartree_factor(energy_unit)
        interpolator, nearest, is_regular, metadata = _make_interpolator(coords_bohr, energies_hartree)
        return cls(
            path=path,
            dims=dims,
            coord_unit=coord_unit,
            energy_unit=energy_unit,
            coord_columns=coord_columns,
            energy_column=energy_column,
            coordinates_bohr=coords_bohr,
            energies_hartree=energies_hartree,
            is_regular_grid=is_regular,
            metadata=metadata,
            _interpolator=interpolator,
            _nearest_interpolator=nearest,
        )

    def evaluate(self, points_bohr: np.ndarray) -> np.ndarray:
        points = np.asarray(points_bohr, dtype=np.float64)
        scalar_input = False
        if self.dims == 1 and points.ndim == 1:
            points = points.reshape(-1, 1)
        elif points.ndim == 1:
            if points.shape[0] != self.dims:
                raise ValueError(f"Expected {self.dims} coordinates, got {points.shape[0]}")
            points = points.reshape(1, self.dims)
            scalar_input = True
        if points.ndim != 2 or points.shape[1] != self.dims:
            raise ValueError(f"Expected points with shape (n, {self.dims}), got {points.shape}")

        if self.dims == 1:
            values = np.asarray(self._interpolator(points[:, 0]), dtype=np.float64)
        else:
            values = np.asarray(self._interpolator(points), dtype=np.float64)
        if np.any(np.isnan(values)):
            if self._nearest_interpolator is None:
                raise ValueError("PES interpolation produced NaN outside the available grid")
            nan_mask = np.isnan(values)
            values[nan_mask] = self._nearest_interpolator(points[nan_mask])
        if scalar_input:
            return values.reshape(1)
        return values.reshape(-1)

    def evaluate_axis(self, mode_index: int, values_bohr: np.ndarray) -> np.ndarray:
        points = np.zeros((len(values_bohr), self.dims), dtype=np.float64)
        points[:, mode_index] = values_bohr
        return self.evaluate(points)

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "dims": self.dims,
            "coord_unit": self.coord_unit,
            "energy_unit": self.energy_unit,
            "coord_columns": self.coord_columns,
            "energy_column": self.energy_column,
            "rows": int(self.coordinates_bohr.shape[0]),
            "is_regular_grid": self.is_regular_grid,
            **self.metadata,
        }


def _read_table(path: Path) -> pd.DataFrame:
    attempts = [
        _try_read_table(path, header="infer", sep=None),
        _try_read_table(path, header=None, sep=None),
        _try_read_table(path, header="infer", sep=r"\s+"),
        _try_read_table(path, header=None, sep=r"\s+"),
    ]
    candidates = [candidate for candidate in attempts if candidate is not None and candidate.shape[1] > 1]
    if not candidates:
        raise ValueError(f"Could not parse numeric PES table: {path}")
    return max(candidates, key=lambda frame: frame.shape[0] * frame.shape[1])


def _try_read_table(path: Path, header: int | str | None, sep: str | None) -> pd.DataFrame | None:
    try:
        frame = pd.read_csv(path, sep=sep, engine="python", header=header)
    except Exception:
        return None
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if numeric.empty or numeric.isna().any().any():
        return None
    numeric.columns = [str(col) for col in frame.columns[: numeric.shape[1]]]
    return numeric


def _make_interpolator(coords: np.ndarray, energies: np.ndarray):
    dims = coords.shape[1]
    if dims == 1:
        order = np.argsort(coords[:, 0])
        x = coords[order, 0]
        y = energies[order]
        return (
            interp1d(x, y, kind="cubic" if len(x) >= 4 else "linear", fill_value="extrapolate"),
            None,
            True,
            {"grid_shape": [int(len(x))], "interpolator": "interp1d"},
        )

    unique_axes = [np.unique(coords[:, axis]) for axis in range(dims)]
    grid_size = int(np.prod([len(axis) for axis in unique_axes]))
    rounded = np.round(coords, decimals=12)
    unique_rows = np.unique(rounded, axis=0).shape[0]
    if grid_size == len(coords) == unique_rows:
        shape = tuple(len(axis) for axis in unique_axes)
        values = np.empty(shape, dtype=np.float64)
        axis_maps = [
            {float(value): idx for idx, value in enumerate(axis_values)}
            for axis_values in unique_axes
        ]
        for point, energy in zip(coords, energies):
            index = tuple(axis_maps[axis][float(point[axis])] for axis in range(dims))
            values[index] = energy
        interpolator = RegularGridInterpolator(
            tuple(unique_axes),
            values,
            bounds_error=False,
            fill_value=None,
        )
        return (
            interpolator,
            None,
            True,
            {"grid_shape": list(shape), "interpolator": "RegularGridInterpolator"},
        )

    linear = LinearNDInterpolator(coords, energies, fill_value=np.nan)
    nearest = NearestNDInterpolator(coords, energies)
    return (
        linear,
        nearest,
        False,
        {"grid_shape": None, "interpolator": "LinearNDInterpolator+NearestNDInterpolator"},
    )
