"""Reference matching and structured output."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import ReferenceSelectionConfig
from .features import extract_query_feature
from .library import ReferenceLibrary, load_reference_library


REQUIRED_LQVE_ASSETS = ("reference_xyz", "modes", "dvr_data", "reference_energies")


@dataclass
class ReferenceSelectionResult:
    query_feature: np.ndarray
    query_metadata: dict[str, Any]
    selected: dict[str, Any]
    candidates: list[dict[str, Any]]
    lqve_ready: bool
    missing_assets: list[str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(
            {
                "query_metadata": self.query_metadata,
                "selected": self.selected,
                "candidates": self.candidates,
                "lqve_ready": self.lqve_ready,
                "missing_assets": self.missing_assets,
                "metadata": self.metadata,
            }
        )

    def save(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "query_feature.npy", self.query_feature.astype(np.float32, copy=False))
        np.savez_compressed(
            output_dir / "reference_match.npz",
            query_feature=self.query_feature.astype(np.float32, copy=False),
            candidate_distances=np.asarray([item["distance"] for item in self.candidates], dtype=np.float64),
            candidate_indices=np.asarray([item["library_index"] for item in self.candidates], dtype=np.int64),
        )
        with (output_dir / "reference_match.json").open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)
        if self.lqve_ready:
            with (output_dir / "selected_reference_config.json").open("w", encoding="utf-8") as handle:
                json.dump(_jsonable(self.selected), handle, indent=2, ensure_ascii=False)


def select_reference(config: ReferenceSelectionConfig) -> ReferenceSelectionResult:
    config.validate()
    query = extract_query_feature(config)
    library = load_reference_library(config.reference_library)
    result = match_query_to_library(
        query_feature=np.asarray(query["feature"], dtype=np.float32),
        query_metadata={key: value for key, value in query.items() if key not in {"feature", "atom_features"}},
        library=library,
        top_k=config.top_k,
        config=config,
    )
    if config.output_dir is not None:
        result.save(config.output_dir)
    return result


def match_query_to_library(
    query_feature: np.ndarray,
    query_metadata: dict[str, Any],
    library: ReferenceLibrary,
    top_k: int,
    config: ReferenceSelectionConfig | None = None,
) -> ReferenceSelectionResult:
    query_feature = np.asarray(query_feature, dtype=np.float32).reshape(-1)
    library.validate_feature_dim(query_feature)
    scaled_library = (library.features - library.mean) / library.std
    scaled_query = (query_feature - library.mean) / library.std
    distances = np.linalg.norm(scaled_library - scaled_query[None, :], axis=1)
    order = np.argsort(distances)[: min(top_k, len(distances))]
    candidates = []
    for rank, lib_idx in enumerate(order, start=1):
        entry = dict(library.entries[int(lib_idx)])
        entry["rank_in_match"] = int(rank)
        entry["library_index"] = int(lib_idx)
        entry["distance"] = float(distances[int(lib_idx)])
        entry["reference_library"] = library.source
        candidates.append(_jsonable(entry))
    selected = dict(candidates[0])
    missing_assets = [key for key in REQUIRED_LQVE_ASSETS if not selected.get(key)]
    metadata = {
        "distance": "standardized_euclidean",
        "reference_library": library.source,
        "reference_feature_type": library.feature_type,
        "reference_checkpoint": library.checkpoint,
        "query_feature_type": "visnet_scalar_x_before_energy_head",
        "top_k": int(top_k),
    }
    if config is not None:
        metadata["config"] = config.to_dict()
    return ReferenceSelectionResult(
        query_feature=query_feature,
        query_metadata=_jsonable(query_metadata),
        selected=_jsonable(selected),
        candidates=candidates,
        lqve_ready=not missing_assets,
        missing_assets=missing_assets,
        metadata=_jsonable(metadata),
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
