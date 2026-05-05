"""Effective-Hamiltonian LQVE shift calculation."""

from __future__ import annotations

import numpy as np

HARTREE_TO_CM1 = 219474.63137054


def normalize_wavefunction_shape(wavefunctions: np.ndarray, grid_count: int, state_count: int) -> np.ndarray:
    wf = np.asarray(wavefunctions, dtype=np.float64)
    if wf.shape == (grid_count, state_count):
        return wf
    if wf.shape == (state_count, grid_count):
        return wf.T
    if wf.ndim == 2 and wf.shape[0] == grid_count:
        return wf
    if wf.ndim == 2 and wf.shape[1] == grid_count:
        return wf.T
    raise ValueError(
        f"Cannot interpret wavefunction shape {wf.shape}; expected grid_count={grid_count}, "
        f"state_count={state_count}"
    )


def build_effective_hamiltonian(
    levels_hartree: np.ndarray,
    wavefunctions: np.ndarray,
    perturbation_hartree: np.ndarray,
    n_contract: int,
) -> np.ndarray:
    levels = np.asarray(levels_hartree, dtype=np.float64)
    wf = normalize_wavefunction_shape(
        wavefunctions,
        grid_count=np.asarray(perturbation_hartree).size,
        state_count=levels.size,
    )
    if n_contract > levels.size or n_contract > wf.shape[1]:
        raise ValueError(
            f"n_contract={n_contract} exceeds available states levels={levels.size}, wf_states={wf.shape[1]}"
        )
    v = np.asarray(perturbation_hartree, dtype=np.float64).reshape(-1)
    if v.size != wf.shape[0]:
        raise ValueError(f"perturbation length {v.size} does not match wavefunction grid {wf.shape[0]}")
    wf_contract = wf[:, :n_contract]
    v_matrix = wf_contract.T @ (v[:, None] * wf_contract)
    return np.diag(levels[:n_contract]) + v_matrix


def solve_effective_levels(
    levels_hartree: np.ndarray,
    wavefunctions: np.ndarray,
    perturbation_hartree: np.ndarray,
    n_contract: int,
) -> np.ndarray:
    hamiltonian = build_effective_hamiltonian(
        levels_hartree=levels_hartree,
        wavefunctions=wavefunctions,
        perturbation_hartree=perturbation_hartree,
        n_contract=n_contract,
    )
    return np.linalg.eigvalsh(hamiltonian)


def transitions_cm1(levels_hartree: np.ndarray, n_transitions: int) -> np.ndarray:
    levels = np.sort(np.asarray(levels_hartree, dtype=np.float64).reshape(-1))
    if levels.size < n_transitions + 1:
        raise ValueError(f"Need at least {n_transitions + 1} levels, got {levels.size}")
    return (levels[1 : n_transitions + 1] - levels[0]) * HARTREE_TO_CM1
