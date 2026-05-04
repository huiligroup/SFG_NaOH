"""Precompute GPUMD-style PBC neighbor lists for the Na12 pickle dataset."""

from __future__ import annotations

import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from visnet.data.neighbors import build_neighbor_list, neighbor_stats
else:
    from .neighbors import build_neighbor_list, neighbor_stats


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "visnet" / "na12_ab.pkl"


def default_edge_path(data_path: Path, cutoff: float) -> Path:
    return data_path.with_name(f"{data_path.stem}_edges_cutoff{cutoff:.1f}.pt")


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
        "cutoff": float(cutoff),
        "max_num_neighbors": int(max_num_neighbors),
        "frame_stride": int(frame_stride),
        "selection_mode": "per_split",
        "splits": sorted(splits),
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
    args = parser.parse_args()
    output = args.output if args.output is not None else default_edge_path(args.data, args.cutoff)
    build_edge_store(
        data_path=args.data,
        output_path=output,
        cutoff=args.cutoff,
        max_num_neighbors=args.max_num_neighbors,
        frame_stride=args.frame_stride,
        splits=set(args.splits),
        limit=args.limit,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
