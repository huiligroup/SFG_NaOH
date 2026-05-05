"""Reference descriptor library loading."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np


@dataclass
class ReferenceLibrary:
    features: np.ndarray
    entries: list[dict[str, Any]]
    mean: np.ndarray
    std: np.ndarray
    source: str
    feature_type: str = "unknown"
    checkpoint: str = ""

    def validate_feature_dim(self, query_feature: np.ndarray) -> None:
        if self.features.ndim != 2:
            raise ValueError("reference features must be a 2D array")
        if query_feature.shape[-1] != self.features.shape[1]:
            raise ValueError(
                f"query feature dim {query_feature.shape[-1]} does not match "
                f"reference dim {self.features.shape[1]}"
            )


def load_reference_library(path: str | Path) -> ReferenceLibrary:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return _load_json_library(path)
    if path.suffix.lower() == ".pkl":
        return _load_pickle_library(path)
    raise ValueError(f"Unsupported reference library file type: {path}")


def _load_pickle_library(path: Path) -> ReferenceLibrary:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if "representatives" in data:
        features = np.asarray(data.get("selected_features"), dtype=np.float32)
        if features.size == 0:
            features = np.stack([np.load(item["feature_file"]) for item in data["representatives"]]).astype(np.float32)
        entries = [dict(item) for item in data["representatives"]]
        for idx, entry in enumerate(entries):
            entry.setdefault("reference_id", _default_reference_id(entry, idx))
            entry.setdefault("feature_file", entry.get("feature_file"))
        mean, std = _mean_std(features, data.get("feature_mean"), data.get("feature_std"))
        return ReferenceLibrary(
            features=features,
            entries=entries,
            mean=mean,
            std=std,
            source=str(path),
            feature_type=str(entries[0].get("feature_type", "unknown")) if entries else "unknown",
            checkpoint=str(entries[0].get("checkpoint", "")) if entries else "",
        )
    if "features" in data:
        features = np.asarray(data["features"], dtype=np.float32)
        entries = []
        for idx in range(features.shape[0]):
            entry = {
                "reference_id": f"record_{idx:08d}",
                "source_record_index": int(idx),
                "mol_id": _array_item(data, "mol_ids", idx),
                "atom_indices": _array_item(data, "atom_indices", idx),
                "traj": _array_item(data, "traj", idx),
                "frame": _array_item(data, "frame", idx),
                "frame_index": _array_item(data, "frame_index", idx),
                "feature_type": data.get("feature_type", "unknown"),
                "checkpoint": data.get("checkpoint", ""),
                "source_data": data.get("source_data", ""),
            }
            entries.append(_jsonable(entry))
        mean, std = _mean_std(features, data.get("feature_mean"), data.get("feature_std"))
        return ReferenceLibrary(
            features=features,
            entries=entries,
            mean=mean,
            std=std,
            source=str(path),
            feature_type=str(data.get("feature_type", "unknown")),
            checkpoint=str(data.get("checkpoint", "")),
        )
    raise ValueError(f"Pickle reference library does not contain features or representatives: {path}")


def _load_json_library(path: Path) -> ReferenceLibrary:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    references = raw if isinstance(raw, list) else raw.get("references")
    if references is None:
        raise ValueError("JSON reference library must be a list or contain a 'references' list")
    entries: list[dict[str, Any]] = []
    features: list[np.ndarray] = []
    for idx, item in enumerate(references):
        entry = dict(item)
        entry.setdefault("reference_id", _default_reference_id(entry, idx))
        if "feature" in entry:
            if isinstance(entry["feature"], str):
                feature = np.load(_resolve_relative(path.parent, entry["feature"])).astype(np.float32)
            else:
                feature = np.asarray(entry["feature"], dtype=np.float32)
        elif "feature_file" in entry:
            feature = np.load(_resolve_relative(path.parent, entry["feature_file"])).astype(np.float32)
        else:
            raise ValueError(f"Reference entry {entry['reference_id']} lacks feature or feature_file")
        features.append(feature.reshape(-1))
        entries.append(entry)
    feature_array = np.stack(features).astype(np.float32, copy=False)
    raw_mean = raw.get("feature_mean") if isinstance(raw, dict) else None
    raw_std = raw.get("feature_std") if isinstance(raw, dict) else None
    mean, std = _mean_std(feature_array, raw_mean, raw_std)
    return ReferenceLibrary(
        features=feature_array,
        entries=entries,
        mean=mean,
        std=std,
        source=str(path),
        feature_type=str(raw.get("feature_type", "unknown")) if isinstance(raw, dict) else "unknown",
        checkpoint=str(raw.get("checkpoint", "")) if isinstance(raw, dict) else "",
    )


def _mean_std(features: np.ndarray, mean: Any = None, std: Any = None) -> tuple[np.ndarray, np.ndarray]:
    if mean is None or std is None:
        mean_array = features.mean(axis=0, dtype=np.float64).astype(np.float32)
        std_array = features.std(axis=0, dtype=np.float64).astype(np.float32)
    else:
        mean_array = np.asarray(mean, dtype=np.float32).reshape(-1)
        std_array = np.asarray(std, dtype=np.float32).reshape(-1)
        if mean_array.size != features.shape[1] or std_array.size != features.shape[1]:
            mean_array = features.mean(axis=0, dtype=np.float64).astype(np.float32)
            std_array = features.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean_array, np.maximum(std_array, 1.0e-8).astype(np.float32)


def _default_reference_id(entry: dict[str, Any], idx: int) -> str:
    if "rank" in entry:
        return f"rank_{int(entry['rank']):04d}"
    if "source_record_index" in entry:
        return f"record_{int(entry['source_record_index']):08d}"
    return f"reference_{idx:04d}"


def _array_item(data: dict[str, Any], key: str, idx: int) -> Any:
    if key not in data:
        return None
    value = np.asarray(data[key], dtype=object)[idx]
    return _jsonable(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _resolve_relative(base: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return base / path
