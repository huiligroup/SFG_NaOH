"""Statistical summaries for LQVE frequency and shift arrays."""

from __future__ import annotations

from typing import Any

import numpy as np


def column_statistics(values: np.ndarray, labels: list[str]) -> list[dict[str, Any]]:
    values = np.asarray(values, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for idx, label in enumerate(labels):
        column = values[:, idx]
        finite = column[np.isfinite(column)]
        row: dict[str, Any] = {
            "label": label,
            "count": int(column.size),
            "finite_count": int(finite.size),
            "nan_count": int(column.size - finite.size),
            "nan_fraction": float((column.size - finite.size) / max(column.size, 1)),
        }
        if finite.size:
            row.update(
                {
                    "mean": float(np.mean(finite)),
                    "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
                    "min": float(np.min(finite)),
                    "q05": float(np.quantile(finite, 0.05)),
                    "median": float(np.median(finite)),
                    "q95": float(np.quantile(finite, 0.95)),
                    "max": float(np.max(finite)),
                }
            )
        else:
            row.update({key: float("nan") for key in ("mean", "std", "min", "q05", "median", "q95", "max")})
        rows.append(row)
    return rows


def frame_quality_rows(
    frame_ids: list[str],
    success: np.ndarray,
    transitions: np.ndarray,
    shifts: np.ndarray,
    source_qc_dir: list[str],
    reference_id: list[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if reference_id is None:
        reference_id = [""] * len(frame_ids)
    for idx, frame_id in enumerate(frame_ids):
        trans_row = transitions[idx]
        shift_row = shifts[idx]
        finite_trans = int(np.isfinite(trans_row).sum())
        finite_shift = int(np.isfinite(shift_row).sum())
        rows.append(
            {
                "frame_index": idx,
                "frame_id": frame_id,
                "success": bool(success[idx]),
                "finite_transition_count": finite_trans,
                "finite_shift_count": finite_shift,
                "transition_nan_count": int(trans_row.size - finite_trans),
                "shift_nan_count": int(shift_row.size - finite_shift),
                "source_qc_dir": source_qc_dir[idx],
                "reference_id": reference_id[idx],
            }
        )
    return rows


def compact_statistics(success: np.ndarray, transitions: np.ndarray, shifts: np.ndarray) -> dict[str, Any]:
    frame_count = int(success.size)
    success_count = int(success.sum())
    return {
        "frame_count": frame_count,
        "success_count": success_count,
        "failure_count": int(frame_count - success_count),
        "success_fraction": float(success_count / max(frame_count, 1)),
        "transition_nan_fraction": float(np.isnan(transitions).sum() / max(transitions.size, 1)),
        "shift_nan_fraction": float(np.isnan(shifts).sum() / max(shifts.size, 1)),
        "transition_count": int(transitions.shape[1]) if transitions.ndim == 2 else 0,
    }


def value_counts(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = value or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return counts
