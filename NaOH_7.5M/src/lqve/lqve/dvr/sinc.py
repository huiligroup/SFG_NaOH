"""Vectorized one-dimensional SINC-DVR and PODVR primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh


@dataclass(frozen=True)
class SincDVRResult:
    levels_hartree: np.ndarray
    wavefunctions: np.ndarray
    grid_bohr: np.ndarray
    hamiltonian_hartree: np.ndarray


@dataclass(frozen=True)
class OneDPODVRResult:
    levels_hartree: np.ndarray
    grid_bohr: np.ndarray
    hamiltonian_hartree: np.ndarray
    wavefunctions: np.ndarray
    sinc: SincDVRResult


def sinc_grid(n_sinc: int, lower_bohr: float, upper_bohr: float) -> np.ndarray:
    if n_sinc < 4:
        raise ValueError("n_sinc must be >= 4")
    step = (upper_bohr - lower_bohr) / n_sinc
    return lower_bohr + step * np.arange(1, n_sinc, dtype=np.float64)


def sinc_kinetic_matrix(
    n_sinc: int,
    lower_bohr: float,
    upper_bohr: float,
    mass_electron: float,
) -> np.ndarray:
    """Build the Colbert-Miller SINC-DVR kinetic matrix in atomic units."""

    if upper_bohr <= lower_bohr:
        raise ValueError("upper_bohr must be larger than lower_bohr")
    if mass_electron <= 0:
        raise ValueError("mass_electron must be positive")
    indices = np.arange(1, n_sinc, dtype=np.float64)
    i = indices[:, None]
    j = indices[None, :]
    prefactor = 1.0 / (2.0 * mass_electron) / (upper_bohr - lower_bohr) ** 2
    matrix = np.empty((n_sinc - 1, n_sinc - 1), dtype=np.float64)

    same = i == j
    matrix[same] = (
        prefactor
        * np.pi**2
        / 2.0
        * ((2.0 * n_sinc**2 + 1.0) / 3.0 - 1.0 / np.sin(np.pi * indices / n_sinc) ** 2)
    )

    diff = i - j
    summ = i + j
    sign = np.where((diff.astype(np.int64) % 2) == 0, 1.0, -1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        off_diag = (
            prefactor
            * sign
            * np.pi**2
            / 2.0
            * (
                1.0 / np.sin(np.pi * diff / (2.0 * n_sinc)) ** 2
                - 1.0 / np.sin(np.pi * summ / (2.0 * n_sinc)) ** 2
            )
        )
    matrix[~same] = off_diag[~same]
    return matrix


def solve_sinc_dvr(
    potential_1d,
    n_sinc: int,
    lower_bohr: float,
    upper_bohr: float,
    mass_electron: float,
) -> SincDVRResult:
    grid = sinc_grid(n_sinc, lower_bohr, upper_bohr)
    potential_values = np.asarray(potential_1d(grid), dtype=np.float64).reshape(-1)
    if potential_values.shape != grid.shape:
        raise ValueError(
            f"1D potential returned shape {potential_values.shape}; expected {grid.shape}"
        )
    hamiltonian = sinc_kinetic_matrix(n_sinc, lower_bohr, upper_bohr, mass_electron)
    hamiltonian = hamiltonian + np.diag(potential_values)
    levels, wavefunctions = eigh(hamiltonian)
    wavefunctions = normalize_wavefunction_signs(wavefunctions)
    return SincDVRResult(levels, wavefunctions, grid, hamiltonian)


def build_1d_podvr(
    potential_1d,
    n_sinc: int,
    n_podvr: int,
    lower_bohr: float,
    upper_bohr: float,
    mass_electron: float,
) -> OneDPODVRResult:
    if n_podvr >= n_sinc:
        raise ValueError("n_podvr must be smaller than n_sinc")
    sinc = solve_sinc_dvr(
        potential_1d=potential_1d,
        n_sinc=n_sinc,
        lower_bohr=lower_bohr,
        upper_bohr=upper_bohr,
        mass_electron=mass_electron,
    )
    states = sinc.wavefunctions[:, :n_podvr]
    position = states.T @ (sinc.grid_bohr[:, None] * states)
    podvr_grid, transform = eigh(position)
    transform = normalize_wavefunction_signs(transform)
    podvr_hamiltonian = transform.T @ np.diag(sinc.levels_hartree[:n_podvr]) @ transform
    podvr_wavefunctions = transform.T
    return OneDPODVRResult(
        levels_hartree=sinc.levels_hartree[:n_podvr],
        grid_bohr=podvr_grid,
        hamiltonian_hartree=podvr_hamiltonian,
        wavefunctions=podvr_wavefunctions,
        sinc=sinc,
    )


def normalize_wavefunction_signs(matrix: np.ndarray) -> np.ndarray:
    """Return a copy with deterministic column signs."""

    result = np.array(matrix, copy=True)
    for col in range(result.shape[1]):
        column = result[:, col]
        idx = int(np.argmax(np.abs(column)))
        if column[idx] < 0:
            result[:, col] *= -1.0
    return result
