"""ViSNet model wrapper with orthorhombic PBC and eIP force uncertainty."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:
    from .data.neighbors import build_neighbor_list, edge_vectors_from_shifts
except ImportError:  # pragma: no cover - supports direct script imports in local experiments.
    from visnet.data.neighbors import build_neighbor_list, edge_vectors_from_shifts


class ExternalPBCDistance(nn.Module):
    """ViSNet distance provider backed by GPUMD-style neighbor lists."""

    def __init__(self, cutoff: float, max_num_neighbors: int = 32, add_self_loops: bool = True) -> None:
        super().__init__()
        self.cutoff = float(cutoff)
        self.max_num_neighbors = int(max_num_neighbors)
        self.add_self_loops = bool(add_self_loops)

    def set_graph(self, cell: Tensor, edge_index: Tensor | None = None, cell_shift: Tensor | None = None) -> None:
        self.cell = cell
        self.edge_index = edge_index
        self.cell_shift = cell_shift

    def forward(self, pos: Tensor, batch: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        cell = getattr(self, "cell", None)
        if cell is None:
            raise RuntimeError("ExternalPBCDistance.cell must be set before calling the representation model")

        edge_index = getattr(self, "edge_index", None)
        cell_shift = getattr(self, "cell_shift", None)
        if edge_index is None or cell_shift is None:
            edge_index, cell_shift = build_neighbor_list(
                pos=pos.detach(),
                batch=batch,
                cell=cell.detach(),
                cutoff=self.cutoff,
                max_num_neighbors=self.max_num_neighbors,
                add_self_loops=self.add_self_loops,
            )
        else:
            edge_index = edge_index.to(device=pos.device, dtype=torch.long)
            cell_shift = cell_shift.to(device=pos.device, dtype=torch.long)

        edge_vec = edge_vectors_from_shifts(pos, edge_index, cell_shift, cell, batch)
        edge_weight = torch.linalg.norm(edge_vec, dim=-1)
        return edge_index, edge_weight, edge_vec


class EvidentialHead(nn.Module):
    def __init__(self, hidden_channels: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.eps = eps
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 3),
        )

    def forward(self, vec_features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if vec_features.dim() != 3 or vec_features.size(1) < 3:
            raise ValueError(f"Expected vector features [N, >=3, H], got {tuple(vec_features.shape)}")
        raw = self.net(vec_features[:, :3, :])
        nu_raw, alpha_raw, beta_raw = raw.unbind(dim=-1)
        nu = F.softplus(nu_raw) + self.eps
        alpha = F.softplus(alpha_raw) + 1.0 + self.eps
        beta = F.softplus(beta_raw) + self.eps
        return nu, alpha, beta


class VisNetEIP(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        num_rbf: int = 32,
        cutoff: float = 5.0,
        max_num_neighbors: int = 64,
        mean: float = 0.0,
        std: float = 1.0,
    ) -> None:
        super().__init__()
        try:
            from torch_geometric.nn.models import ViSNet
            from torch_geometric.utils import scatter
        except ImportError as exc:
            raise ImportError(
                "VisNetEIP requires torch_geometric. Install it with `pip install torch_geometric`."
            ) from exc
        self.scatter = scatter
        self.backbone = ViSNet(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            num_rbf=num_rbf,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            derivative=False,
            reduce_op="sum",
            mean=mean,
            std=std,
        )
        self.pbc_distance = ExternalPBCDistance(
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            add_self_loops=True,
        )
        self.backbone.representation_model.distance = self.pbc_distance
        self.evidential_head = EvidentialHead(hidden_channels=hidden_channels)

    def _representation(self, z: Tensor, pos: Tensor, batch: Tensor) -> tuple[Tensor, Tensor]:
        """Run PyG's ViSNetBlock with the edge-vector normalization made functional.

        PyG 2.7 normalizes ``edge_vec`` with an in-place indexed assignment.
        That is fine for its built-in distance implementation in many settings,
        but conflicts with differentiating energy through the PBC minimum-image
        vectors used here. The rest of this method mirrors ViSNetBlock.forward.
        """

        rep = self.backbone.representation_model
        x = rep.embedding(z)
        edge_index, edge_weight, edge_vec = rep.distance(pos, batch)
        edge_attr = rep.distance_expansion(edge_weight)
        mask = edge_index[0] != edge_index[1]
        norm = torch.linalg.norm(edge_vec, dim=1).clamp_min(1.0e-12)
        edge_unit = torch.where(mask.unsqueeze(1), edge_vec / norm.unsqueeze(1), edge_vec)
        edge_sh = rep.sphere(edge_unit)
        x = rep.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr)
        vec = torch.zeros(
            x.size(0),
            ((rep.lmax + 1) ** 2) - 1,
            x.size(1),
            dtype=x.dtype,
            device=x.device,
        )
        edge_attr = rep.edge_embedding(edge_index, edge_attr, x)

        for attn in rep.vis_mp_layers[:-1]:
            dx, dvec, dedge_attr = attn(x, vec, edge_index, edge_weight, edge_attr, edge_sh)
            x = x + dx
            vec = vec + dvec
            edge_attr = edge_attr + dedge_attr

        dx, dvec, _ = rep.vis_mp_layers[-1](x, vec, edge_index, edge_weight, edge_attr, edge_sh)
        x = x + dx
        vec = vec + dvec

        x = rep.out_norm(x)
        vec = rep.vec_out_norm(vec)
        return x, vec

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Tensor,
        cell: Tensor,
        edge_index: Tensor | None = None,
        cell_shift: Tensor | None = None,
    ) -> dict[str, Tensor]:
        pos = pos.requires_grad_(True)
        self.pbc_distance.set_graph(cell=cell, edge_index=edge_index, cell_shift=cell_shift)
        x, v = self._representation(z, pos, batch)
        atom_energy = self.backbone.output_model.pre_reduce(x, v)
        atom_energy = atom_energy * self.backbone.std
        if self.backbone.prior_model is not None:
            atom_energy = self.backbone.prior_model(atom_energy, z)
        energy = self.scatter(atom_energy, batch, dim=0, reduce=self.backbone.reduce_op)
        energy = energy + self.backbone.mean
        grad = torch.autograd.grad(
            [energy],
            [pos],
            grad_outputs=[torch.ones_like(energy)],
            create_graph=self.training,
            retain_graph=True,
        )[0]
        if grad is None:
            raise RuntimeError("Autograd returned None for force prediction")
        force = -grad
        nu, alpha, beta = self.evidential_head(v)
        epistemic = beta / (nu * (alpha - 1.0)).clamp_min(1.0e-8)
        aleatoric = beta / (alpha - 1.0).clamp_min(1.0e-8)
        return {
            "energy": energy.view(-1),
            "force": force,
            "nu": nu,
            "alpha": alpha,
            "beta": beta,
            "epistemic": epistemic,
            "aleatoric": aleatoric,
        }
