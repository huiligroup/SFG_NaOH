"""Dataset and collation utilities for the Na12 pickle dataset."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .neighbors import build_neighbor_list
except ImportError:  # pragma: no cover
    from visnet.data.neighbors import build_neighbor_list


class NaOHDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        split: str = "train",
        limit: int | None = None,
        frame_stride: int = 4,
        edge_store: dict | None = None,
        cutoff: float | None = None,
        max_num_neighbors: int | None = None,
        fallback_missing_edges: bool = True,
    ) -> None:
        if frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")
        self.path = Path(path)
        with self.path.open("rb") as handle:
            self.data = pickle.load(handle)
        mask = np.asarray(self.data["split"]) == split
        self.indices = np.flatnonzero(mask)[::frame_stride]
        if limit is not None:
            self.indices = self.indices[:limit]
        if len(self.indices) == 0:
            raise ValueError(f"No frames for split={split!r} in {self.path}")
        self.z = torch.as_tensor(self.data["z"], dtype=torch.long)
        self.natoms = int(self.z.numel())
        self.split = split
        self.frame_stride = frame_stride
        self.edge_store = edge_store
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.fallback_missing_edges = fallback_missing_edges
        self.edge_lookup = None
        if edge_store is not None:
            self.edge_lookup = {int(frame_idx): i for i, frame_idx in enumerate(edge_store["frame_indices"])}

    @property
    def energy_mean_std(self) -> tuple[float, float]:
        values = np.asarray(self.data["energy"], dtype=np.float64)[self.indices]
        std = float(values.std())
        return float(values.mean()), max(std, 1.0e-8)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        frame_idx = int(self.indices[idx])
        sample = {
            "z": self.z,
            "pos": torch.as_tensor(self.data["pos"][frame_idx], dtype=torch.float32),
            "energy": torch.as_tensor([self.data["energy"][frame_idx]], dtype=torch.float32),
            "forces": torch.as_tensor(self.data["forces"][frame_idx], dtype=torch.float32),
            "cell": torch.as_tensor(self.data["cell"][frame_idx], dtype=torch.float32),
            "traj": str(self.data["traj"][frame_idx]),
            "frame": int(self.data["frame"][frame_idx]),
            "frame_index": frame_idx,
        }
        if self.edge_store is not None and self.edge_lookup is not None:
            if frame_idx in self.edge_lookup:
                edge_item = self.edge_store["entries"][self.edge_lookup[frame_idx]]
                sample["edge_index"] = edge_item["edge_index"].to(torch.long)
                sample["cell_shift"] = edge_item["cell_shift"].to(torch.long)
            elif self.fallback_missing_edges and self.cutoff is not None and self.max_num_neighbors is not None:
                edge_index, cell_shift = build_neighbor_list(
                    pos=sample["pos"],
                    batch=torch.zeros(self.natoms, dtype=torch.long),
                    cell=sample["cell"].unsqueeze(0),
                    cutoff=self.cutoff,
                    max_num_neighbors=self.max_num_neighbors,
                    add_self_loops=True,
                )
                sample["edge_index"] = edge_index
                sample["cell_shift"] = cell_shift
            else:
                raise KeyError(
                    f"Frame index {frame_idx} is missing from precomputed edge data. "
                    "Regenerate edge data with the current precompute_edges.py, or enable fallback_missing_edges."
                )
        return sample


def collate_frames(samples: list[dict[str, torch.Tensor | str | int]]) -> dict[str, torch.Tensor | list[str] | list[int]]:
    z_parts = []
    pos_parts = []
    force_parts = []
    energy_parts = []
    batch_parts = []
    cell_parts = []
    edge_parts = []
    shift_parts = []
    traj_parts: list[str] = []
    frame_parts: list[int] = []
    frame_index_parts: list[int] = []
    offset = 0
    for batch_idx, sample in enumerate(samples):
        z = sample["z"]
        pos = sample["pos"]
        forces = sample["forces"]
        energy = sample["energy"]
        cell = sample["cell"]
        if not isinstance(z, torch.Tensor) or not isinstance(pos, torch.Tensor):
            raise TypeError("collate_frames expects tensor samples")
        natoms = int(z.numel())
        z_parts.append(z)
        pos_parts.append(pos)
        force_parts.append(forces)
        energy_parts.append(energy)
        batch_parts.append(torch.full((natoms,), batch_idx, dtype=torch.long))
        cell_parts.append(cell)
        if "edge_index" in sample:
            edge_index = sample["edge_index"]
            cell_shift = sample["cell_shift"]
            if not isinstance(edge_index, torch.Tensor) or not isinstance(cell_shift, torch.Tensor):
                raise TypeError("edge_index and cell_shift must be tensors")
            edge_parts.append(edge_index + offset)
            shift_parts.append(cell_shift)
        traj_parts.append(str(sample["traj"]))
        frame_parts.append(int(sample["frame"]))
        frame_index_parts.append(int(sample["frame_index"]))
        offset += natoms
    batch = {
        "z": torch.cat(z_parts, dim=0),
        "pos": torch.cat(pos_parts, dim=0),
        "forces": torch.cat(force_parts, dim=0),
        "energy": torch.cat(energy_parts, dim=0),
        "batch": torch.cat(batch_parts, dim=0),
        "cell": torch.stack(cell_parts, dim=0),
        "traj": traj_parts,
        "frame": frame_parts,
        "frame_index": frame_index_parts,
    }
    if edge_parts:
        if len(edge_parts) != len(samples):
            raise ValueError("Either all samples or no samples must include precomputed edges")
        batch["edge_index"] = torch.cat(edge_parts, dim=1)
        batch["cell_shift"] = torch.cat(shift_parts, dim=0)
    return batch
