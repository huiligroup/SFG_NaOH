"""Precompute GPUMD-style PBC neighbor lists for a ViSNet pickle dataset."""

from __future__ import annotations

import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from visnet.data.neighbors import build_neighbor_list, neighbor_stats
else:
    from .neighbors import build_neighbor_list, neighbor_stats


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "visnet" / "naoh12.pkl"


def default_edge_path(data_path: Path, cutoff: float) -> Path:
    return data_path.with_name(f"{data_path.stem}_edges_cutoff{cutoff:.1f}.pt")


def _normalized_splits(splits: set[str]) -> list[str]:
    return sorted(splits)


def _resolve_loose(path_like: str | Path | None) -> str | None:
    if path_like is None:
        return None
    return str(Path(path_like).expanduser().resolve(strict=False))


def dataset_signature(data_path: Path) -> dict[str, Any]:
    stat = data_path.stat()
    return {
        "resolved_data_path": _resolve_loose(data_path),
        "data_size_bytes": int(stat.st_size),
        "data_mtime_ns": int(stat.st_mtime_ns),
    }


def selected_indices(dataset: dict, splits: set[str], frame_stride: int, limit: int | None) -> np.ndarray:
    split_values = np.asarray(dataset["split"])
    split_order = [name for name in ("train", "val", "test") if name in splits]
    split_order.extend(sorted(splits.difference(split_order)))
    parts = [np.flatnonzero(split_values == split)[::frame_stride] for split in split_order]
    indices = np.concatenate(parts) if parts else np.asarray([], dtype=np.int64)
    if limit is not None:
        indices = indices[:limit]
    return indices


def _build_one_edge(args: tuple[int, np.ndarray, np.ndarray, int, float, int]) -> dict[str, torch.Tensor]:
    frame_idx, pos_array, cell_array, natoms, cutoff, max_num_neighbors = args
    pos = torch.as_tensor(pos_array, dtype=torch.float32)
    cell = torch.as_tensor(cell_array, dtype=torch.float32).unsqueeze(0)
    batch = torch.zeros(natoms, dtype=torch.long)
    edge_index, cell_shift = build_neighbor_list(
        pos=pos,
        batch=batch,
        cell=cell,
        cutoff=cutoff,
        max_num_neighbors=max_num_neighbors,
        add_self_loops=True,
    )
    return {
        "frame_index": torch.tensor(int(frame_idx), dtype=torch.long),
        "edge_index": edge_index.cpu().to(torch.int32),
        "cell_shift": cell_shift.cpu().to(torch.int8),
    }


def build_edge_store(
    data_path: Path,
    output_path: Path,
    cutoff: float,
    max_num_neighbors: int,
    frame_stride: int,
    splits: set[str],
    limit: int | None = None,
    num_workers: int = 1,
) -> dict:
    with data_path.open("rb") as handle:
        dataset = pickle.load(handle)
    indices = selected_indices(dataset, splits=splits, frame_stride=frame_stride, limit=limit)
    signature = dataset_signature(data_path)
    z = torch.as_tensor(dataset["z"], dtype=torch.long)
    natoms = int(z.numel())
    tasks = (
        (
            int(frame_idx),
            np.asarray(dataset["pos"][frame_idx], dtype=np.float32),
            np.asarray(dataset["cell"][frame_idx], dtype=np.float32),
            natoms,
            cutoff,
            max_num_neighbors,
        )
        for frame_idx in indices
    )
    entries: list[dict[str, torch.Tensor]] = []
    if num_workers <= 1:
        iterator = map(_build_one_edge, tasks)
        for count, entry in enumerate(iterator, start=1):
            entries.append(entry)
            if count % 100 == 0:
                print(f"built_edges frames={count}/{len(indices)}")
    else:
        try:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                for count, entry in enumerate(executor.map(_build_one_edge, tasks, chunksize=8), start=1):
                    entries.append(entry)
                    if count % 100 == 0:
                        print(f"built_edges frames={count}/{len(indices)}")
        except PermissionError as exc:
            print(f"parallel_edge_build_unavailable={exc}; falling back to num_workers=1")
            tasks = (
                (
                    int(frame_idx),
                    np.asarray(dataset["pos"][frame_idx], dtype=np.float32),
                    np.asarray(dataset["cell"][frame_idx], dtype=np.float32),
                    natoms,
                    cutoff,
                    max_num_neighbors,
                )
                for frame_idx in indices
            )
            for count, entry in enumerate(map(_build_one_edge, tasks), start=1):
                entries.append(entry)
                if count % 100 == 0:
                    print(f"built_edges frames={count}/{len(indices)}")

    stats = neighbor_stats(entries, natoms=natoms)
    store = {
        "data_path": str(data_path),
        "resolved_data_path": signature["resolved_data_path"],
        "data_size_bytes": signature["data_size_bytes"],
        "data_mtime_ns": signature["data_mtime_ns"],
        "cutoff": float(cutoff),
        "max_num_neighbors": int(max_num_neighbors),
        "frame_stride": int(frame_stride),
        "selection_mode": "per_split",
        "splits": _normalized_splits(splits),
        "limit": None if limit is None else int(limit),
        "num_workers": int(num_workers),
        "natoms": natoms,
        "frame_indices": [int(idx) for idx in indices],
        "entries": entries,
        "stats": {
            "frames": stats.frames,
            "edges": stats.edges,
            "atoms": stats.atoms,
            "average_neighbors": stats.average_neighbors,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(store, output_path)
    print(f"wrote_edge_data={output_path}")
    print(f"frames={stats.frames} avg_neighbors={stats.average_neighbors:.3f}")
    return store


def edge_store_cache_status(
    store: dict,
    *,
    data_path: Path,
    cutoff: float,
    max_num_neighbors: int,
    frame_stride: int,
    splits: set[str],
    limit: int | None,
) -> tuple[bool, str]:
    requested_signature = dataset_signature(data_path)
    expected_splits = _normalized_splits(splits)
    expected_limit = None if limit is None else int(limit)

    if float(store.get("cutoff", float("nan"))) != float(cutoff):
        return False, f"cutoff_mismatch store={store.get('cutoff')} requested={cutoff}"
    if int(store.get("max_num_neighbors", -1)) != int(max_num_neighbors):
        return False, (
            f"max_num_neighbors_mismatch store={store.get('max_num_neighbors')} "
            f"requested={max_num_neighbors}"
        )
    if int(store.get("frame_stride", -1)) != int(frame_stride):
        return False, f"frame_stride_mismatch store={store.get('frame_stride')} requested={frame_stride}"
    if list(store.get("splits", [])) != expected_splits:
        return False, f"splits_mismatch store={store.get('splits')} requested={expected_splits}"
    if store.get("selection_mode", "per_split") != "per_split":
        return False, f"selection_mode_mismatch store={store.get('selection_mode')}"
    if store.get("limit") != expected_limit:
        return False, f"limit_mismatch store={store.get('limit')} requested={expected_limit}"

    stored_resolved = store.get("resolved_data_path")
    if stored_resolved is not None and stored_resolved != requested_signature["resolved_data_path"]:
        return False, (
            f"data_path_mismatch store={stored_resolved} "
            f"requested={requested_signature['resolved_data_path']}"
        )
    if "data_size_bytes" in store and int(store["data_size_bytes"]) != requested_signature["data_size_bytes"]:
        return False, (
            f"data_size_mismatch store={store['data_size_bytes']} "
            f"requested={requested_signature['data_size_bytes']}"
        )
    if "data_mtime_ns" in store and int(store["data_mtime_ns"]) != requested_signature["data_mtime_ns"]:
        return False, (
            f"data_mtime_mismatch store={store['data_mtime_ns']} "
            f"requested={requested_signature['data_mtime_ns']}"
        )

    with data_path.open("rb") as handle:
        dataset = pickle.load(handle)
    expected_indices = [int(idx) for idx in selected_indices(dataset, splits=splits, frame_stride=frame_stride, limit=limit)]
    stored_indices = [int(idx) for idx in store.get("frame_indices", [])]
    if stored_indices != expected_indices:
        return False, (
            f"frame_indices_mismatch store_count={len(stored_indices)} requested_count={len(expected_indices)}"
        )

    natoms = int(np.asarray(dataset["z"]).shape[0])
    if int(store.get("natoms", -1)) != natoms:
        return False, f"natoms_mismatch store={store.get('natoms')} requested={natoms}"

    if "data_size_bytes" not in store or "data_mtime_ns" not in store or "resolved_data_path" not in store:
        return True, "legacy_cache_match"
    return True, "cache_match"


def load_edge_store_if_valid(
    *,
    edge_path: Path,
    data_path: Path,
    cutoff: float,
    max_num_neighbors: int,
    frame_stride: int,
    splits: set[str],
    limit: int | None,
) -> tuple[dict | None, str]:
    if not edge_path.exists():
        return None, "cache_missing"
    store = torch.load(edge_path, map_location="cpu", weights_only=False)
    valid, reason = edge_store_cache_status(
        store,
        data_path=data_path,
        cutoff=cutoff,
        max_num_neighbors=max_num_neighbors,
        frame_stride=frame_stride,
        splits=splits,
        limit=limit,
    )
    if not valid:
        return None, reason
    return store, reason


def ensure_edge_store(
    *,
    data_path: Path,
    output_path: Path,
    cutoff: float,
    max_num_neighbors: int,
    frame_stride: int,
    splits: set[str],
    limit: int | None = None,
    num_workers: int = 1,
    force_rebuild: bool = False,
) -> tuple[dict, bool, str]:
    if not force_rebuild:
        store, reason = load_edge_store_if_valid(
            edge_path=output_path,
            data_path=data_path,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            frame_stride=frame_stride,
            splits=splits,
            limit=limit,
        )
        if store is not None:
            print(
                f"reused_edge_data={output_path} reason={reason} frames={len(store['frame_indices'])} "
                f"avg_neighbors={store.get('stats', {}).get('average_neighbors', float('nan')):.3f}"
            )
            return store, True, reason
        if output_path.exists():
            print(f"edge_data_stale={output_path} reason={reason}; rebuilding")

    store = build_edge_store(
        data_path=data_path,
        output_path=output_path,
        cutoff=cutoff,
        max_num_neighbors=max_num_neighbors,
        frame_stride=frame_stride,
        splits=splits,
        limit=limit,
        num_workers=num_workers,
    )
    return store, False, "rebuilt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--max-num-neighbors", type=int, default=64)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--force-rebuild", action="store_true", help="Ignore an existing compatible edge cache.")
    args = parser.parse_args()
    output = args.output if args.output is not None else default_edge_path(args.data, args.cutoff)
    ensure_edge_store(
        data_path=args.data,
        output_path=output,
        cutoff=args.cutoff,
        max_num_neighbors=args.max_num_neighbors,
        frame_stride=args.frame_stride,
        splits=set(args.splits),
        limit=args.limit,
        num_workers=args.num_workers,
        force_rebuild=args.force_rebuild,
    )


if __name__ == "__main__":
    main()
