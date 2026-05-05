"""LQVE frequency-shift calculation from DVR and QC energies."""

from __future__ import annotations

from typing import Any

import numpy as np

from lqve.dvr.result import DVRResult

from .config import ShiftConfig
from .effective_hamiltonian import solve_effective_levels, transitions_cm1
from .result import ShiftResult


def compute_shift(
    dvr_result: DVRResult,
    qc_output: dict[str, Any],
    reference_energies_hartree: np.ndarray,
    config: ShiftConfig,
) -> ShiftResult:
    levels = np.asarray(dvr_result.levels_hartree, dtype=np.float64).reshape(-1)
    wavefunctions = np.asarray(dvr_result.wavefunctions, dtype=np.float64)
    reference = np.asarray(reference_energies_hartree, dtype=np.float64).reshape(-1)
    qc_energies = np.asarray(qc_output["energies_hartree"], dtype=np.float64).reshape(-1)
    grid_points = np.asarray(qc_output["grid_points"], dtype=np.float64)
    success_mask = np.asarray(qc_output["success_mask"], dtype=bool).reshape(-1)
    source_indices = qc_output.get("source_indices")
    grid_count = int(np.prod(dvr_result.basis_shape))
    expected_grid_points = _dvr_grid_points(dvr_result)

    _validate_inputs(
        levels=levels,
        wavefunctions=wavefunctions,
        reference=reference,
        qc_energies=qc_energies,
        success_mask=success_mask,
        grid_count=grid_count,
        config=config,
    )
    _validate_grid_contract(grid_points, expected_grid_points, source_indices)
    if source_indices is not None and not np.array_equal(np.asarray(source_indices, dtype=np.int64), np.arange(grid_count)):
        raise ValueError("QC source_indices must cover the full DVR grid in order after sorting")
    if source_indices is None and grid_points.shape != expected_grid_points.shape:
        raise ValueError(
            "QC grid_points do not match DVR grid shape; "
            f"qc_shape={grid_points.shape}, dvr_shape={expected_grid_points.shape}"
        )
    reference_transitions = transitions_cm1(levels, config.n_transitions)
    frame_id = str(qc_output["frame_id"])
    if not bool(np.all(success_mask)) or not np.all(np.isfinite(qc_energies)):
        nan_transitions = np.full(config.n_transitions, np.nan, dtype=np.float64)
        return ShiftResult(
            frame_id=frame_id,
            levels_hartree=np.full(config.n_contract, np.nan, dtype=np.float64),
            transitions_cm1=nan_transitions,
            shifts_cm1=nan_transitions.copy(),
            success_mask=success_mask,
            grid_points=grid_points,
            source_qc_dir=str(qc_output["qc_dir"]),
            metadata={
                "reason": "qc_output_contains_failed_or_nonfinite_grid_points",
                "reference_transitions_cm1": reference_transitions.tolist(),
            },
        )

    perturbation = qc_energies - reference
    if config.subtract_mean:
        perturbation = perturbation - float(np.mean(perturbation))
    effective_levels = solve_effective_levels(
        levels_hartree=levels,
        wavefunctions=wavefunctions,
        perturbation_hartree=perturbation,
        n_contract=config.n_contract,
    )
    transitions = transitions_cm1(effective_levels, config.n_transitions)
    return ShiftResult(
        frame_id=frame_id,
        levels_hartree=effective_levels,
        transitions_cm1=transitions,
        shifts_cm1=transitions - reference_transitions,
        success_mask=success_mask,
        grid_points=grid_points,
        source_qc_dir=str(qc_output["qc_dir"]),
        metadata={
            "method": "effective_hamiltonian",
            "n_contract": config.n_contract,
            "n_transitions": config.n_transitions,
            "subtract_mean": config.subtract_mean,
            "reference_transitions_cm1": reference_transitions.tolist(),
            "perturbation_min_hartree": float(np.min(perturbation)),
            "perturbation_max_hartree": float(np.max(perturbation)),
        },
    )


def _validate_inputs(
    levels: np.ndarray,
    wavefunctions: np.ndarray,
    reference: np.ndarray,
    qc_energies: np.ndarray,
    success_mask: np.ndarray,
    grid_count: int,
    config: ShiftConfig,
) -> None:
    if reference.size != grid_count:
        raise ValueError(f"reference energy length {reference.size} does not match DVR grid count {grid_count}")
    if qc_energies.size != grid_count:
        raise ValueError(f"QC energy length {qc_energies.size} does not match DVR grid count {grid_count}")
    if success_mask.size != grid_count:
        raise ValueError(f"QC success_mask length {success_mask.size} does not match DVR grid count {grid_count}")
    if wavefunctions.ndim != 2:
        raise ValueError(f"wavefunctions must be 2D, got shape {wavefunctions.shape}")
    if config.n_contract > levels.size:
        raise ValueError(f"n_contract={config.n_contract} exceeds DVR state count {levels.size}")
    if config.n_transitions >= config.n_contract:
        raise ValueError("n_transitions must be smaller than n_contract")


def _dvr_grid_points(dvr_result: DVRResult) -> np.ndarray:
    mesh = np.meshgrid(*[np.asarray(grid, dtype=np.float64) for grid in dvr_result.grids_bohr], indexing="ij")
    return np.stack([axis.reshape(-1) for axis in mesh], axis=1)


def _validate_grid_contract(
    grid_points: np.ndarray,
    expected_grid_points_bohr: np.ndarray,
    source_indices: np.ndarray | None,
) -> None:
    if grid_points.shape[0] != expected_grid_points_bohr.shape[0]:
        raise ValueError(
            "QC grid count does not match DVR grid count: "
            f"qc={grid_points.shape[0]}, dvr={expected_grid_points_bohr.shape[0]}"
        )
    if source_indices is not None:
        return
    if grid_points.shape != expected_grid_points_bohr.shape:
        raise ValueError(
            "QC grid_points do not match DVR grid shape and source_indices are missing: "
            f"qc_shape={grid_points.shape}, dvr_shape={expected_grid_points_bohr.shape}"
        )
    bohr_to_angstrom = 0.52917721092
    if np.allclose(grid_points, expected_grid_points_bohr, atol=1.0e-8):
        return
    if np.allclose(grid_points, expected_grid_points_bohr * bohr_to_angstrom, atol=1.0e-8):
        return
    raise ValueError("QC grid_points do not match DVR grid values in Bohr or Angstrom")
