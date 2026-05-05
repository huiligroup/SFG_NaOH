"""I/O helpers for LQVE result analysis."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def resolve_shift_summary(shifts: str | Path) -> Path:
    path = Path(shifts)
    if path.is_file() and path.name == "shifts_summary.csv":
        return path
    candidate = path / "shifts_summary.csv"
    if candidate.exists():
        return candidate
    if path.is_dir():
        children = sorted(child / "shifts_summary.csv" for child in path.iterdir() if (child / "shifts_summary.csv").exists())
        if children:
            return children[0]
    raise FileNotFoundError(f"No shifts_summary.csv found under {path}")


def load_shift_summary(shifts: str | Path) -> dict[str, Any]:
    summary_csv = resolve_shift_summary(shifts)
    rows = _read_csv_rows(summary_csv)
    if not rows:
        raise ValueError(f"Shift summary is empty: {summary_csv}")
    transition_cols = _numbered_columns(rows[0], "transition_", "_cm1")
    shift_cols = _numbered_columns(rows[0], "shift_", "_cm1")
    if not transition_cols or not shift_cols:
        raise ValueError("shifts_summary.csv must contain transition_N_cm1 and shift_N_cm1 columns")
    if len(transition_cols) != len(shift_cols):
        raise ValueError("transition and shift column counts differ")
    frame_ids = [row.get("frame_id", str(idx)) for idx, row in enumerate(rows)]
    success = np.asarray([_parse_bool(row.get("success", "False")) for row in rows], dtype=bool)
    source_qc_dir = [row.get("source_qc_dir", "") for row in rows]
    reference_id = [row.get("reference_id", "") for row in rows]
    transitions = _rows_to_array(rows, transition_cols)
    shifts_arr = _rows_to_array(rows, shift_cols)
    _check_npz_if_present(summary_csv.parent, transitions, shifts_arr, success)
    return {
        "summary_csv": summary_csv,
        "shift_dir": summary_csv.parent,
        "frame_ids": frame_ids,
        "success": success,
        "source_qc_dir": source_qc_dir,
        "reference_id": reference_id,
        "transitions_cm1": transitions,
        "shifts_cm1": shifts_arr,
        "transition_columns": transition_cols,
        "shift_columns": shift_cols,
        "rows": rows,
    }


def select_transition_indices(available_count: int, transitions: list[int] | None) -> list[int]:
    if transitions is None:
        return list(range(available_count))
    indices = [item - 1 for item in transitions]
    invalid = [item + 1 for item in indices if item < 0 or item >= available_count]
    if invalid:
        raise ValueError(f"Requested transition indices out of range: {invalid}; available=1..{available_count}")
    return indices


def write_table(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _numbered_columns(row: dict[str, str], prefix: str, suffix: str) -> list[str]:
    columns = [key for key in row.keys() if key.startswith(prefix) and key.endswith(suffix)]
    return sorted(columns, key=lambda key: int(key[len(prefix) : -len(suffix)]))


def _rows_to_array(rows: list[dict[str, str]], columns: list[str]) -> np.ndarray:
    data = np.full((len(rows), len(columns)), np.nan, dtype=np.float64)
    for row_idx, row in enumerate(rows):
        for col_idx, column in enumerate(columns):
            data[row_idx, col_idx] = _parse_float(row.get(column, "nan"))
    return data


def _parse_float(value: str | float | int | None) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"true", "1", "yes", "y"}


def _check_npz_if_present(shift_dir: Path, transitions: np.ndarray, shifts: np.ndarray, success: np.ndarray) -> None:
    npz_path = shift_dir / "shifts_summary.npz"
    if not npz_path.exists():
        return
    with np.load(npz_path, allow_pickle=True) as data:
        if "transitions_cm1" in data and data["transitions_cm1"].shape != transitions.shape:
            raise ValueError("shifts_summary.npz transitions_cm1 shape does not match CSV")
        if "shifts_cm1" in data and data["shifts_cm1"].shape != shifts.shape:
            raise ValueError("shifts_summary.npz shifts_cm1 shape does not match CSV")
        if "success" in data and data["success"].shape != success.shape:
            raise ValueError("shifts_summary.npz success shape does not match CSV")
