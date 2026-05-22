"""Candidate novelty checks and farthest-point selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import CandidateFrame


@dataclass(frozen=True)
class SelectionSummary:
    selected_indices: list[int]
    selected_ids: list[str]
    distances: list[float]


def _standardize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(features, dtype=np.float64)
    mean = array.mean(axis=0)
    std = array.std(axis=0)
    std = np.where(std < 1.0e-8, 1.0, std)
    return (array - mean) / std, mean, std


def is_candidate_novel(
    feature: np.ndarray,
    existing: list[CandidateFrame],
    min_distance: float,
) -> bool:
    if not existing:
        return True
    matrix = np.stack([item.frame_feature for item in existing] + [np.asarray(feature, dtype=np.float32)], axis=0)
    scaled, _, _ = _standardize(matrix)
    probe = scaled[-1]
    dists = np.linalg.norm(scaled[:-1] - probe[None, :], axis=1)
    return bool(np.all(dists >= min_distance))


def select_representatives(candidates: list[CandidateFrame], k: int) -> SelectionSummary:
    if not candidates:
        raise ValueError("No candidates provided for representative selection")
    if k < 1:
        raise ValueError("k must be >= 1")
    features = np.stack([item.frame_feature for item in candidates], axis=0)
    scaled, _, _ = _standardize(features)
    if len(candidates) == 1:
        return SelectionSummary(selected_indices=[0], selected_ids=[candidates[0].candidate_id], distances=[0.0])

    diff = scaled[:, None, :] - scaled[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)
    mean_dist = dist_matrix.mean(axis=1)
    first = int(np.argmax(mean_dist))
    selected = [first]
    selected_distances = [float(mean_dist[first])]
    min_dist_to_selected = dist_matrix[:, first].copy()

    while len(selected) < min(k, len(candidates)):
        min_dist_to_selected[selected] = -np.inf
        next_idx = int(np.argmax(min_dist_to_selected))
        if not np.isfinite(min_dist_to_selected[next_idx]):
            break
        selected.append(next_idx)
        selected_distances.append(float(min_dist_to_selected[next_idx]))
        min_dist_to_selected = np.minimum(min_dist_to_selected, dist_matrix[:, next_idx])

    return SelectionSummary(
        selected_indices=selected,
        selected_ids=[candidates[idx].candidate_id for idx in selected],
        distances=selected_distances,
    )
