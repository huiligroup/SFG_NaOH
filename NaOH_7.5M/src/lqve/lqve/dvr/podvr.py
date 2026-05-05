"""Generic N-dimensional PODVR construction."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy import sparse
from scipy.sparse.linalg import eigsh

from .config import DVRConfig
from .potential import PotentialSurface
from .result import DVRResult
from .sinc import OneDPODVRResult, build_1d_podvr, normalize_wavefunction_signs


def build_reference_dvr(
    pes_path: str | Path,
    config: DVRConfig,
) -> DVRResult:
    """Build an N-dimensional reference PODVR result from a PES table."""

    config.validate()
    start = time.perf_counter()
    potential = PotentialSurface.from_file(
        pes_path,
        dims=config.dims,
        coord_unit=config.coord_unit,
        energy_unit=config.energy_unit,
    )
    ranges_bohr = config.ranges_bohr()
    one_d_results: list[OneDPODVRResult] = []
    for mode_index, mode in enumerate(config.modes):
        lower_bohr, upper_bohr = ranges_bohr[mode_index]
        one_d_results.append(
            build_1d_podvr(
                potential_1d=lambda x, idx=mode_index: potential.evaluate_axis(idx, np.asarray(x)),
                n_sinc=mode.sinc_points,
                n_podvr=mode.podvr_points,
                lower_bohr=lower_bohr,
                upper_bohr=upper_bohr,
                mass_electron=mode.mass,
            )
        )

    hamiltonian = _build_sparse_hamiltonian(one_d_results)
    coupling_diag = _evaluate_coupling_diagonal(potential, one_d_results)
    hamiltonian = hamiltonian + sparse.diags(coupling_diag, offsets=0, format="csr")
    levels, wavefunctions, solver_name, used_sparse = _solve_lowest_states(hamiltonian, config)
    levels = np.real(levels)
    wavefunctions = normalize_wavefunction_signs(np.real(wavefunctions))
    order = np.argsort(levels)
    levels = levels[order]
    wavefunctions = wavefunctions[:, order]
    elapsed = time.perf_counter() - start

    metadata = {
        "config": config.to_dict(),
        "potential": potential.metadata_dict(),
        "solver": solver_name,
        "used_sparse_solver": used_sparse,
        "hamiltonian_shape": list(hamiltonian.shape),
        "hamiltonian_nnz": int(hamiltonian.nnz),
        "coupling_min_hartree": float(np.min(coupling_diag)),
        "coupling_max_hartree": float(np.max(coupling_diag)),
        "elapsed_seconds": elapsed,
        "one_d": [
            {
                "mode": config.modes[idx].name,
                "grid_size": int(len(result.grid_bohr)),
                "lowest_level_hartree": float(result.levels_hartree[0]),
            }
            for idx, result in enumerate(one_d_results)
        ],
    }
    return DVRResult(
        levels_hartree=levels[: config.states],
        wavefunctions=wavefunctions[:, : config.states],
        grids_bohr=[result.grid_bohr for result in one_d_results],
        basis_shape=config.basis_shape,
        metadata=metadata,
    )


def _build_sparse_hamiltonian(results: list[OneDPODVRResult]) -> sparse.csr_matrix:
    dims = len(results)
    basis_shape = [len(result.grid_bohr) for result in results]
    total = sparse.csr_matrix((int(np.prod(basis_shape)), int(np.prod(basis_shape))))
    identities = [sparse.identity(size, format="csr", dtype=np.float64) for size in basis_shape]
    for mode_index, result in enumerate(results):
        term = None
        for axis in range(dims):
            factor = (
                sparse.csr_matrix(result.hamiltonian_hartree)
                if axis == mode_index
                else identities[axis]
            )
            term = factor if term is None else sparse.kron(term, factor, format="csr")
        total = total + term
    return total.tocsr()


def _evaluate_coupling_diagonal(
    potential: PotentialSurface,
    results: list[OneDPODVRResult],
) -> np.ndarray:
    grids = [result.grid_bohr for result in results]
    mesh = np.meshgrid(*grids, indexing="ij")
    points = np.stack([axis.reshape(-1) for axis in mesh], axis=1)
    total = potential.evaluate(points)
    separable = np.zeros_like(total)
    for mode_index, grid in enumerate(grids):
        shape = [1] * len(grids)
        shape[mode_index] = len(grid)
        values = potential.evaluate_axis(mode_index, grid).reshape(shape)
        separable += np.broadcast_to(values, tuple(len(g) for g in grids)).reshape(-1)
    return total - separable


def _solve_lowest_states(
    hamiltonian: sparse.csr_matrix,
    config: DVRConfig,
) -> tuple[np.ndarray, np.ndarray, str, bool]:
    size = hamiltonian.shape[0]
    states = config.states
    if size <= config.dense_threshold or states >= size - 1:
        dense = hamiltonian.toarray()
        levels, wavefunctions = eigh(dense, subset_by_index=[0, states - 1])
        return levels, wavefunctions, "scipy.linalg.eigh", False
    levels, wavefunctions = eigsh(
        hamiltonian,
        k=states,
        which="SA",
        tol=config.solver_tol,
    )
    return levels, wavefunctions, "scipy.sparse.linalg.eigsh", True

