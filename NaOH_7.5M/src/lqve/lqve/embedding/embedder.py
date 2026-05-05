"""Vectorized arbitrary-dimensional DVR embedding."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .alignment import kabsch_align
from .config import EmbeddingConfig
from .io import (
    convert_grids_to_angstrom,
    convert_modes_to_internal,
    infer_masses,
    read_dvr_grids,
    read_modes,
    read_xyz,
)
from .result import EmbeddingResult


def build_embedded_geometries(config: EmbeddingConfig) -> EmbeddingResult:
    config.validate()
    start = time.perf_counter()
    ref_species, reference_positions, _ = read_xyz(config.reference_xyz, frame_index=0)
    system_species, system_positions, _ = read_xyz(config.system_xyz, frame_index=config.frame_index)
    probe_indices = np.asarray(config.probe_indices, dtype=np.int64)
    if np.max(probe_indices) >= len(system_species):
        raise ValueError("probe_indices exceed system atom count")
    probe_positions = system_positions[probe_indices]
    if len(ref_species) != len(probe_indices):
        raise ValueError(
            f"reference probe atom count {len(ref_species)} does not match "
            f"probe_indices count {len(probe_indices)}"
        )
    probe_species = [system_species[idx] for idx in probe_indices]
    if probe_species != ref_species:
        raise ValueError(
            "Probe species order does not match reference species: "
            f"reference={ref_species}, probe={probe_species}"
        )

    masses = (
        np.asarray(config.masses, dtype=np.float64)
        if config.masses is not None
        else infer_masses(ref_species)
    )
    modes = convert_modes_to_internal(read_modes(config.modes, len(ref_species)), config.mode_unit)
    grids = convert_grids_to_angstrom(read_dvr_grids(config.dvr_data), config.grid_unit)
    embedded_probe, grid_points, alignment, projection = embed_dvr_geometries(
        reference_positions=reference_positions,
        probe_positions=probe_positions,
        masses=masses,
        modes=modes,
        grids=grids,
        centroid_weight=config.centroid_weight,
    )
    embedded_system = np.broadcast_to(system_positions, (len(grid_points), *system_positions.shape)).copy()
    embedded_system[:, probe_indices, :] = embedded_probe
    elapsed = time.perf_counter() - start
    metadata = {
        "reference_xyz": str(config.reference_xyz),
        "system_xyz": str(config.system_xyz),
        "modes": str(config.modes),
        "dvr_data": str(config.dvr_data),
        "frame_index": config.frame_index,
        "grid_unit": config.grid_unit,
        "mode_unit": config.mode_unit,
        "coord_unit": config.coord_unit,
        "centroid_weight": config.centroid_weight,
        "dims": int(len(grids)),
        "grid_sizes": [int(len(grid)) for grid in grids],
        "grid_count": int(len(grid_points)),
        "mode_shape": list(modes.shape),
        "probe_atom_count": int(len(ref_species)),
        "system_atom_count": int(len(system_species)),
        "probe_indices": probe_indices.tolist(),
        "species": system_species,
        "reference_species": ref_species,
        "write_xyz": bool(config.write_xyz),
        "prefix": config.prefix,
        "alignment_rmsd_angstrom": alignment.rmsd,
        "projection_coefficients": projection["coefficients"].tolist(),
        "projected_mode_overlaps": projection["residual_overlaps"].tolist(),
        "elapsed_seconds": elapsed,
        **config.metadata,
    }
    return EmbeddingResult(
        embedded_probe=embedded_probe,
        embedded_system=embedded_system,
        grid_points=grid_points,
        probe_indices=probe_indices,
        reference_positions=reference_positions,
        system_positions=system_positions,
        rotation=alignment.rotation_to_reference,
        translation=alignment.mobile_centroid,
        species=system_species,
        metadata=metadata,
    )


def embed_dvr_geometries(
    reference_positions: np.ndarray,
    probe_positions: np.ndarray,
    masses: np.ndarray,
    modes: np.ndarray,
    grids: list[np.ndarray],
    centroid_weight: str = "mass",
) -> tuple[np.ndarray, np.ndarray, object, dict[str, np.ndarray]]:
    reference_positions = np.asarray(reference_positions, dtype=np.float64)
    probe_positions = np.asarray(probe_positions, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    modes = np.asarray(modes, dtype=np.float64)
    if modes.ndim != 3 or modes.shape[1:] != reference_positions.shape:
        raise ValueError(
            f"modes must have shape (D, N, 3), got {modes.shape}; "
            f"expected (*, {reference_positions.shape[0]}, 3)"
        )
    if len(grids) != modes.shape[0]:
        raise ValueError(f"Number of grids {len(grids)} does not match modes {modes.shape[0]}")

    alignment = kabsch_align(reference_positions, probe_positions, masses, centroid_weight)
    delta = alignment.aligned_mobile - reference_positions
    delta_perp, projection = project_modes_out(delta, modes, masses)
    grid_points = make_grid_points(grids)
    embedded_reference = (
        reference_positions[None, :, :]
        + delta_perp[None, :, :]
        + np.einsum("gd,dnc->gnc", grid_points, modes, optimize=True)
    )
    embedded_probe = (
        (embedded_reference - alignment.reference_centroid[None, None, :])
        @ alignment.rotation_to_reference.T
        + alignment.mobile_centroid[None, None, :]
    )
    return embedded_probe, grid_points, alignment, projection


def project_modes_out(
    delta: np.ndarray,
    modes: np.ndarray,
    masses: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    sqrt_m = np.sqrt(masses)[:, None]
    mw_delta = delta * sqrt_m
    mw_modes = modes * sqrt_m[None, :, :]
    flat_delta = mw_delta.reshape(-1)
    flat_modes = mw_modes.reshape(modes.shape[0], -1)
    gram = flat_modes @ flat_modes.T
    mode_norms = np.diag(gram)
    if np.any(mode_norms <= 0):
        raise ValueError("Normal modes must have non-zero mass-weighted norm")
    rhs = flat_modes @ flat_delta
    coefficients = np.linalg.solve(gram, rhs)
    residual = flat_delta - np.einsum("d,di->i", coefficients, flat_modes, optimize=True)
    residual_overlaps = np.einsum("i,di->d", residual, flat_modes, optimize=True)
    delta_perp = residual.reshape(delta.shape) / sqrt_m
    return delta_perp, {
        "coefficients": coefficients,
        "residual_overlaps": residual_overlaps,
        "mode_norms": mode_norms,
        "mode_gram": gram,
    }


def make_grid_points(grids: list[np.ndarray]) -> np.ndarray:
    if not grids:
        raise ValueError("At least one DVR grid is required")
    mesh = np.meshgrid(*[np.asarray(grid, dtype=np.float64) for grid in grids], indexing="ij")
    return np.stack([axis.reshape(-1) for axis in mesh], axis=1)
