"""Select representative molecule structures from extracted ViSNet descriptors."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURES = ROOT / "src" / "descriptor" / "mol_features.pkl"
DEFAULT_OUTPUT_DIR = ROOT / "src" / "descriptor" / "represent_mol"


def jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def standardize(features: np.ndarray, eps: float = 1.0e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = features.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, eps)
    return ((features - mean) / std).astype(np.float32, copy=False), mean, std


def exact_mean_distances(features: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    total = tensor.size(0)
    sums = torch.empty(total, dtype=torch.float64, device="cpu")
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        block = tensor[start:stop]
        block_sum = torch.zeros(stop - start, dtype=torch.float64, device=device)
        for other_start in range(0, total, batch_size):
            other_stop = min(other_start + batch_size, total)
            dist = torch.cdist(block, tensor[other_start:other_stop])
            block_sum += dist.sum(dim=1, dtype=torch.float64)
        sums[start:stop] = block_sum.cpu()
        print(f"mean_distance_blocks={stop}/{total}")
    return (sums.numpy() / max(total - 1, 1)).astype(np.float32)


def centroid_distances(features: np.ndarray) -> np.ndarray:
    centroid = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    return np.linalg.norm(features - centroid, axis=1).astype(np.float32)


def update_min_distances(
    features: np.ndarray,
    selected_feature: np.ndarray,
    current: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    selected = torch.as_tensor(selected_feature[None, :], dtype=torch.float32, device=device)
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        block = torch.as_tensor(features[start:stop], dtype=torch.float32, device=device)
        dist = torch.cdist(block, selected).view(-1).cpu().numpy().astype(np.float32)
        current[start:stop] = np.minimum(current[start:stop], dist)
    return current


def farthest_point_sample(
    features: np.ndarray,
    k: int,
    initial_mode: str,
    batch_size: int,
    device: torch.device,
) -> tuple[list[int], np.ndarray]:
    if k < 1:
        raise ValueError("--k must be >= 1")
    if k > len(features):
        raise ValueError(f"--k={k} is larger than available records={len(features)}")
    if initial_mode == "exact-mean":
        initial_scores = exact_mean_distances(features, batch_size=batch_size, device=device)
    elif initial_mode == "centroid":
        initial_scores = centroid_distances(features)
    else:
        raise ValueError(f"Unknown initial mode: {initial_mode}")

    selected = [int(np.argmax(initial_scores))]
    min_dist = np.full(len(features), np.inf, dtype=np.float32)
    min_dist = update_min_distances(features, features[selected[0]], min_dist, batch_size, device)
    min_dist[selected[0]] = -np.inf

    while len(selected) < k:
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        min_dist = update_min_distances(features, features[next_idx], min_dist, batch_size, device)
        min_dist[selected] = -np.inf
        print(f"selected={len(selected)}/{k} record_index={next_idx}")
    return selected, initial_scores


def write_xyz(path: Path, species: np.ndarray, pos: np.ndarray, metadata: dict[str, Any]) -> None:
    comment = (
        f"traj={metadata['traj']} frame={metadata['frame']} frame_index={metadata['frame_index']} "
        f"mol_id={metadata['mol_id']} rank={metadata['rank']}"
    )
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(species)}\n")
        handle.write(comment + "\n")
        for symbol, xyz in zip(species, pos):
            handle.write(f"{str(symbol):2s} {xyz[0]: .10f} {xyz[1]: .10f} {xyz[2]: .10f}\n")


def record_metadata(data: dict, source_index: int, rank: int, score: float, min_distance: float | None) -> dict[str, Any]:
    frame_pos_index = int(data["record_frame_pos_index"][source_index])
    return {
        "rank": int(rank),
        "source_record_index": int(source_index),
        "mol_id": int(data["mol_ids"][source_index]),
        "atom_indices": np.asarray(data["atom_indices"][source_index], dtype=np.int64).tolist(),
        "mol_species": np.asarray(data["mol_species"][source_index], dtype=object).tolist(),
        "traj": str(data["traj"][source_index]),
        "frame": int(data["frame"][source_index]),
        "frame_index": int(data["frame_index"][source_index]),
        "system_frame_pos_index": frame_pos_index,
        "initial_score": float(score),
        "nearest_selected_distance": None if min_distance is None else float(min_distance),
        "feature_type": str(data.get("feature_type", "unknown")),
        "source_data": str(data.get("source_data", "")),
        "checkpoint": str(data.get("checkpoint", "")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--mol-id", type=int, default=None, help="Filter to one molecule id before selection.")
    parser.add_argument("--all-mols", action="store_true", help="Use all molecule ids in the feature file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--distance-batch-size", type=int, default=512)
    parser.add_argument("--initial-mode", choices=("exact-mean", "centroid"), default="exact-mean")
    args = parser.parse_args()

    if args.distance_batch_size < 1:
        raise ValueError("--distance-batch-size must be >= 1")
    with args.features.open("rb") as handle:
        data = pickle.load(handle)

    features = np.asarray(data["features"], dtype=np.float32)
    if args.mol_id is not None and not args.all_mols:
        mask = np.asarray(data["mol_ids"]) == args.mol_id
    else:
        mask = np.ones(len(features), dtype=bool)
    source_indices = np.flatnonzero(mask)
    if len(source_indices) == 0:
        raise ValueError("No descriptor records match the requested molecule filter")
    filtered_features = features[source_indices]
    scaled_features, feature_mean, feature_std = standardize(filtered_features)

    device = torch.device(args.device)
    selected_local, initial_scores = farthest_point_sample(
        scaled_features,
        k=args.k,
        initial_mode=args.initial_mode,
        batch_size=args.distance_batch_size,
        device=device,
    )
    selected_source = [int(source_indices[idx]) for idx in selected_local]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    representatives: list[dict[str, Any]] = []
    selected_scaled = scaled_features[selected_local]
    min_distances: list[float | None] = [None]
    if len(selected_scaled) > 1:
        for idx in range(1, len(selected_scaled)):
            dist = np.linalg.norm(selected_scaled[idx] - selected_scaled[:idx], axis=1)
            min_distances.append(float(dist.min()))

    for rank, (local_idx, source_idx, min_distance) in enumerate(
        zip(selected_local, selected_source, min_distances), start=1
    ):
        metadata = record_metadata(
            data=data,
            source_index=source_idx,
            rank=rank,
            score=float(initial_scores[local_idx]),
            min_distance=min_distance,
        )
        folder = args.output_dir / (
            f"rank_{rank:04d}_mol_{metadata['mol_id']:03d}_"
            f"traj_{metadata['traj']}_frame_{metadata['frame']:08d}"
        )
        folder.mkdir(parents=True, exist_ok=True)
        feature = np.asarray(data["features"][source_idx], dtype=np.float32)
        np.save(folder / "feature.npy", feature)
        frame_pos_index = int(data["record_frame_pos_index"][source_idx])
        write_xyz(folder / "system.xyz", np.asarray(data["species"], dtype=object), data["system_pos"][frame_pos_index], metadata)
        with (folder / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(jsonable(metadata), handle, ensure_ascii=False, indent=2)
        representatives.append(metadata | {"folder": str(folder), "feature_file": str(folder / "feature.npy")})

    summary = {
        "feature_file": str(args.features),
        "output_dir": str(args.output_dir),
        "k": int(args.k),
        "mol_id_filter": None if args.all_mols else args.mol_id,
        "initial_mode": args.initial_mode,
        "distance": "standardized_euclidean",
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "selected_source_indices": np.asarray(selected_source, dtype=np.int64),
        "selected_features": features[selected_source].astype(np.float32, copy=False),
        "representatives": representatives,
    }
    with (args.output_dir / "representatives.pkl").open("wb") as handle:
        pickle.dump(summary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote_representatives={args.output_dir} selected={len(representatives)}")


if __name__ == "__main__":
    main()
