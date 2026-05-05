"""I/O helpers for LQVE shift calculations."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def resolve_qc_output_dirs(qc_outputs: str | Path) -> list[Path]:
    path = Path(qc_outputs)
    if path.is_file() and path.name == "qc_energies.npz":
        return [path.parent]
    if (path / "qc_energies.npz").exists():
        return [path]
    if path.is_dir():
        children = sorted(child for child in path.iterdir() if (child / "qc_energies.npz").exists())
        if children:
            return children
    raise FileNotFoundError(f"No qc_energies.npz found under {path}")


def load_qc_output(qc_dir: str | Path) -> dict[str, Any]:
    qc_dir = Path(qc_dir)
    with np.load(qc_dir / "qc_energies.npz") as data:
        arrays = {key: np.array(data[key]) for key in data.files}
    metadata_path = qc_dir / "metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    energies = np.asarray(arrays["energies_hartree"], dtype=np.float64).reshape(-1)
    grid_points = np.asarray(arrays["grid_points"], dtype=np.float64)
    success_mask = np.asarray(arrays["success_mask"], dtype=bool).reshape(-1)
    source_indices = arrays.get("source_indices")
    if source_indices is not None:
        order = np.argsort(np.asarray(source_indices, dtype=np.int64).reshape(-1))
        energies = energies[order]
        grid_points = grid_points[order]
        success_mask = success_mask[order]
        source_indices = np.asarray(source_indices, dtype=np.int64).reshape(-1)[order]
    return {
        "qc_dir": qc_dir,
        "frame_id": str(metadata.get("frame_id", qc_dir.name)),
        "energies_hartree": energies,
        "grid_points": grid_points,
        "success_mask": success_mask,
        "source_indices": source_indices,
        "metadata": metadata,
    }


def load_reference_energies(path: str | Path, key: str | None = None) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as data:
            if key is not None:
                if key not in data:
                    raise KeyError(f"Reference key {key!r} not found in {path}; available={list(data.files)}")
                array = data[key]
            elif "energies_hartree" in data:
                array = data["energies_hartree"]
            elif "reference_energies_hartree" in data:
                array = data["reference_energies_hartree"]
            else:
                array = data[list(data.files)[0]]
    elif suffix == ".csv":
        array = _load_text_vector(path, delimiter=",", key=key)
    elif suffix in {".txt", ".dat"}:
        array = _load_text_vector(path, delimiter=None, key=key)
    else:
        raise ValueError(f"Unsupported reference energy file type: {path}")
    array = np.asarray(array, dtype=np.float64)
    if array.ndim > 1:
        array = array[:, -1]
    return array.reshape(-1)


def _load_text_vector(path: Path, delimiter: str | None, key: str | None = None) -> np.ndarray:
    if key is not None:
        data = np.genfromtxt(path, delimiter=delimiter, names=True)
        if not data.dtype.names or key not in data.dtype.names:
            raise KeyError(f"Reference column {key!r} not found in {path}")
        return np.asarray(data[key], dtype=np.float64)
    try:
        return np.loadtxt(path, delimiter=delimiter)
    except ValueError:
        data = np.genfromtxt(path, delimiter=delimiter, names=True)
        if data.dtype.names:
            return np.asarray(data[data.dtype.names[-1]], dtype=np.float64)
        return np.asarray(data, dtype=np.float64)


def write_summary_csv(path: str | Path, rows: list[dict[str, Any]], n_transitions: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["frame_id", "success", "source_qc_dir", "reference_id", "dvr_data", "reference_energies"]
    fieldnames += [f"transition_{idx + 1}_cm1" for idx in range(n_transitions)]
    fieldnames += [f"shift_{idx + 1}_cm1" for idx in range(n_transitions)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_manifest(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
