"""Alignment helpers for embedding DVR geometries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AlignmentResult:
    aligned_mobile: np.ndarray
    rotation_to_reference: np.ndarray
    reference_centroid: np.ndarray
    mobile_centroid: np.ndarray
    weights: np.ndarray
    rmsd: float


def weighted_centroid(positions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {positions.shape}")
    if weights.shape != (positions.shape[0],):
        raise ValueError("weights must have shape (N,)")
    total = np.sum(weights)
    if total <= 0:
        raise ValueError("centroid weights must sum to a positive value")
    return np.sum(positions * weights[:, None], axis=0) / total


def kabsch_align(
    reference: np.ndarray,
    mobile: np.ndarray,
    masses: np.ndarray,
    centroid_weight: str = "mass",
) -> AlignmentResult:
    """Align ``mobile`` to ``reference`` with a weighted Kabsch rotation.

    Coordinates use row vectors. ``rotation_to_reference`` maps centered mobile
    coordinates into the centered reference frame via ``mobile @ R``.
    """

    reference = np.asarray(reference, dtype=np.float64)
    mobile = np.asarray(mobile, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    if reference.shape != mobile.shape:
        raise ValueError(f"reference and mobile shape mismatch: {reference.shape} vs {mobile.shape}")
    if reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("reference and mobile must have shape (N, 3)")
    weights = centroid_weights(masses, centroid_weight)
    ref_centroid = weighted_centroid(reference, weights)
    mob_centroid = weighted_centroid(mobile, weights)
    ref_centered = reference - ref_centroid
    mob_centered = mobile - mob_centroid
    rotation = kabsch_rotation(mob_centered, ref_centered, weights)
    aligned_centered = mob_centered @ rotation
    aligned = aligned_centered + ref_centroid
    rmsd = float(np.sqrt(np.average(np.sum((aligned - reference) ** 2, axis=1), weights=weights)))
    return AlignmentResult(
        aligned_mobile=aligned,
        rotation_to_reference=rotation,
        reference_centroid=ref_centroid,
        mobile_centroid=mob_centroid,
        weights=weights,
        rmsd=rmsd,
    )


def centroid_weights(masses: np.ndarray, mode: str) -> np.ndarray:
    masses = np.asarray(masses, dtype=np.float64)
    if masses.ndim != 1 or np.any(masses <= 0):
        raise ValueError("masses must be a positive one-dimensional array")
    normalized = mode.lower()
    if normalized == "mass":
        return masses
    if normalized == "sqrt_mass":
        return np.sqrt(masses)
    if normalized == "uniform":
        return np.ones_like(masses)
    raise ValueError(f"Unsupported centroid weight mode: {mode}")


def kabsch_rotation(mobile_centered: np.ndarray, reference_centered: np.ndarray, weights: np.ndarray) -> np.ndarray:
    covariance = (mobile_centered * weights[:, None]).T @ reference_centered
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(u @ vt)) or 1.0
    return u @ correction @ vt

