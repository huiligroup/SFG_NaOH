"""GPUMD-style orthorhombic PBC neighbor-list utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class NeighborStats:
    frames: int
    edges: int
    atoms: int

    @property
    def average_neighbors(self) -> float:
        return self.edges / max(self.atoms, 1)


def cell_lengths(cell: Tensor) -> Tensor:
    if cell.dim() == 3:
        return torch.diagonal(cell, dim1=-2, dim2=-1)
    if cell.dim() == 2 and cell.shape[-1] == 3:
        return cell
    raise ValueError(f"Expected cell shape [B, 3, 3] or [B, 3], got {tuple(cell.shape)}")


def build_neighbor_list(
    pos: Tensor,
    batch: Tensor,
    cell: Tensor,
    cutoff: float,
    max_num_neighbors: int,
    add_self_loops: bool = True,
) -> tuple[Tensor, Tensor]:
    """Build ``edge_index`` and integer image shifts outside autograd.

    The returned convention matches PyG ViSNet's distance direction:
    ``edge_vec = pos[src] + cell_shift @ cell - pos[dst]``.
    """

    with torch.no_grad():
        lengths = cell_lengths(cell).to(device=pos.device, dtype=pos.dtype)
        edge_parts: list[Tensor] = []
        shift_parts: list[Tensor] = []
        for graph_idx in torch.unique(batch, sorted=True):
            atom_idx = torch.nonzero(batch == graph_idx, as_tuple=False).view(-1)
            local_pos = pos[atom_idx]
            box = lengths[int(graph_idx.item())].view(1, 1, 3)
            raw_diff = local_pos[:, None, :] - local_pos[None, :, :]
            image_shift = -torch.round(raw_diff / box).to(torch.long)
            diff = raw_diff + image_shift.to(raw_diff.dtype) * box
            dist = torch.linalg.norm(diff, dim=-1)

            for dst in range(local_pos.size(0)):
                src_local = torch.nonzero((dist[:, dst] < cutoff) & (dist[:, dst] > 0.0), as_tuple=False).view(-1)
                if src_local.numel() > max_num_neighbors:
                    nearest = torch.topk(dist[src_local, dst], k=max_num_neighbors, largest=False).indices
                    src_local = src_local[nearest]
                if src_local.numel() > 0:
                    dst_local = torch.full_like(src_local, dst)
                    edge_parts.append(torch.stack([atom_idx[src_local], atom_idx[dst_local]], dim=0))
                    shift_parts.append(image_shift[src_local, dst])

            if add_self_loops:
                edge_parts.append(torch.stack([atom_idx, atom_idx], dim=0))
                shift_parts.append(torch.zeros((atom_idx.numel(), 3), dtype=torch.long, device=pos.device))

        if not edge_parts:
            return (
                torch.empty((2, 0), dtype=torch.long, device=pos.device),
                torch.empty((0, 3), dtype=torch.long, device=pos.device),
            )
        return torch.cat(edge_parts, dim=1), torch.cat(shift_parts, dim=0)


def edge_vectors_from_shifts(pos: Tensor, edge_index: Tensor, cell_shift: Tensor, cell: Tensor, batch: Tensor) -> Tensor:
    src, dst = edge_index
    edge_cell = cell[batch[dst]]
    shift_vec = torch.bmm(cell_shift.to(dtype=pos.dtype).unsqueeze(1), edge_cell.to(dtype=pos.dtype)).squeeze(1)
    return pos[src] + shift_vec - pos[dst]


def neighbor_stats(edge_entries: list[dict[str, Tensor]], natoms: int) -> NeighborStats:
    edges = sum(int(item["edge_index"].shape[1]) for item in edge_entries)
    return NeighborStats(frames=len(edge_entries), edges=edges, atoms=max(len(edge_entries) * natoms, 1))

