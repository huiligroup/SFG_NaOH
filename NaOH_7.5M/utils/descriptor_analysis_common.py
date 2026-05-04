"""Shared utilities for ViSNet descriptor analysis scripts."""

from __future__ import annotations

import csv
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = ROOT / "src" / "descriptor" / "mol_features.pkl"
DEFAULT_REPRESENTATIVES = ROOT / "src" / "descriptor" / "represent_mol" / "representatives.pkl"
DEFAULT_REPORT_DIR = ROOT / "utils" / "descriptor_report"

ELEMENT_COLORS = {
    "H": "#d9d9d9",
    "O": "#d62728",
    "Na": "#1f77b4",
}


def configure_matplotlib() -> None:
    cache_root = Path(tempfile.gettempdir()) / "naoh_visnet_descriptor"
    mpl_dir = cache_root / "matplotlib"
    cache_dir = cache_root / "cache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg", force=True)


def load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def dump_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(data), handle, ensure_ascii=False, indent=2)


def dump_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: to_jsonable(row.get(key, "")) for key in fieldnames})


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def standardize(features: np.ndarray, eps: float = 1.0e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float32)
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = features.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, eps)
    scaled = ((features - mean) / std).astype(np.float32, copy=False)
    return scaled, mean, std


def representative_source_indices(representatives: dict[str, Any]) -> np.ndarray:
    if "selected_source_indices" in representatives:
        return np.asarray(representatives["selected_source_indices"], dtype=np.int64)
    rows = representatives.get("representatives", [])
    return np.asarray([int(row["source_record_index"]) for row in rows], dtype=np.int64)


def representative_rows(representatives: dict[str, Any], selected_indices: np.ndarray) -> list[dict[str, Any]]:
    rows = list(representatives.get("representatives", []))
    if rows:
        return rows
    return [{"rank": i + 1, "source_record_index": int(idx)} for i, idx in enumerate(selected_indices)]


def cdist_minibatch(
    points: np.ndarray,
    centers: np.ndarray,
    batch_size: int = 4096,
) -> np.ndarray:
    """Return distance from each point to each center using bounded memory."""

    points = np.asarray(points, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    out = np.empty((len(points), len(centers)), dtype=np.float32)
    center_sq = np.sum(centers * centers, axis=1)[None, :]
    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        block = points[start:stop]
        dist_sq = np.sum(block * block, axis=1)[:, None] + center_sq - 2.0 * block @ centers.T
        out[start:stop] = np.sqrt(np.maximum(dist_sq, 0.0))
    return out


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    dist_sq = np.sum(points * points, axis=1)[:, None] + np.sum(points * points, axis=1)[None, :] - 2.0 * points @ points.T
    return np.sqrt(np.maximum(dist_sq, 0.0))


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "mean": float(np.mean(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def minimum_image_vectors(vectors: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Apply minimum-image convention for row-vector coordinates."""

    vectors = np.asarray(vectors, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    inv_cell = np.linalg.inv(cell)
    frac = vectors @ inv_cell
    return vectors - np.round(frac) @ cell


def pbc_distances(points: np.ndarray, reference_points: np.ndarray, cell: np.ndarray) -> np.ndarray:
    diff = np.asarray(points, dtype=np.float64)[:, None, :] - np.asarray(reference_points, dtype=np.float64)[None, :, :]
    mic = minimum_image_vectors(diff.reshape(-1, 3), cell).reshape(diff.shape)
    return np.linalg.norm(mic, axis=-1)


def pbc_distance(a: np.ndarray, b: np.ndarray, cell: np.ndarray) -> float:
    return float(np.linalg.norm(minimum_image_vectors(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64), cell)))


def pbc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray, cell: np.ndarray) -> float:
    """Angle ABC in degrees under PBC."""

    ba = minimum_image_vectors(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64), cell)
    bc = minimum_image_vectors(np.asarray(c, dtype=np.float64) - np.asarray(b, dtype=np.float64), cell)
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom <= 0.0:
        return float("nan")
    cos_value = float(np.dot(ba, bc) / denom)
    return float(np.degrees(np.arccos(np.clip(cos_value, -1.0, 1.0))))


def record_frame_position(features: dict[str, Any], source_index: int) -> tuple[np.ndarray, np.ndarray]:
    frame_pos_index = int(features["record_frame_pos_index"][source_index])
    pos = np.asarray(features["system_pos"][frame_pos_index], dtype=np.float32)
    if "system_cell" in features:
        cell = np.asarray(features["system_cell"][frame_pos_index], dtype=np.float32)
    else:
        cell = np.asarray(features["cell"][source_index], dtype=np.float32)
    return pos, cell


def record_metadata(features: dict[str, Any], source_index: int, rank: int | None = None) -> dict[str, Any]:
    row = {
        "source_record_index": int(source_index),
        "mol_id": int(features["mol_ids"][source_index]),
        "traj": str(features["traj"][source_index]),
        "frame": int(features["frame"][source_index]),
        "frame_index": int(features["frame_index"][source_index]),
        "atom_indices": np.asarray(features["atom_indices"][source_index], dtype=np.int64).tolist(),
        "mol_species": np.asarray(features["mol_species"][source_index], dtype=object).tolist(),
    }
    if rank is not None:
        row["rank"] = int(rank)
    return row
