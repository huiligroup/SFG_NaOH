"""ViSNet checkpoint backend for full-box LQVE energy evaluation."""

from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from .base import EnergyBackend
from lqve.qc.io import atomic_numbers
from lqve.qc.types import EnergyResult, GeometryBatch


class VisNetBackend(EnergyBackend):
    name = "visnet"

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cpu",
        batch_size: int = 16,
        cell: np.ndarray | None = None,
        **options: Any,
    ) -> None:
        super().__init__(
            checkpoint="" if checkpoint is None else str(checkpoint),
            device=device,
            batch_size=batch_size,
            cell=None if cell is None else np.asarray(cell).tolist(),
            **options,
        )
        if checkpoint is None:
            raise ValueError("VisNet backend requires a checkpoint path")
        self.checkpoint = Path(checkpoint)
        self.device_name = device
        self.batch_size = int(batch_size)
        self.default_cell = None if cell is None else np.asarray(cell, dtype=np.float64)
        self._torch = None
        self._model = None

    def evaluate(self, batch: GeometryBatch) -> EnergyResult:
        torch, model = self._load_model()
        frame_cell = batch.cell if batch.cell is not None else self.default_cell
        if frame_cell is None:
            raise ValueError("VisNet backend requires a periodic cell from embedding metadata or --cell")
        if not batch.species:
            raise ValueError("VisNet backend requires non-empty species from embedded geometry metadata")
        z_single = torch.as_tensor(atomic_numbers(batch.species), dtype=torch.long, device=self.device)
        energies: list[np.ndarray] = []
        start = time.perf_counter()
        with torch.no_grad():
            for start_idx in range(0, batch.geometry_count, self.batch_size):
                sub = batch.slice(start_idx, min(start_idx + self.batch_size, batch.geometry_count))
                pos = torch.as_tensor(sub.positions.reshape(-1, 3), dtype=torch.float32, device=self.device)
                batch_index = torch.repeat_interleave(
                    torch.arange(sub.geometry_count, dtype=torch.long, device=self.device),
                    repeats=sub.atom_count,
                )
                z = z_single.repeat(sub.geometry_count)
                sub_cell = sub.cell if sub.cell is not None else frame_cell
                cell_tensor = self._cell_tensor(torch, sub_cell, sub.geometry_count)
                energy = self._energy_only(model, z, pos, batch_index, cell_tensor)
                energies.append(energy.detach().cpu().numpy().astype(np.float64))
        energy_array = np.concatenate(energies) if energies else np.empty(0, dtype=np.float64)
        return EnergyResult(
            backend=self.name,
            frame_id=batch.frame_id,
            energies_hartree=energy_array,
            grid_points=batch.grid_points,
            success_mask=np.isfinite(energy_array),
            source_indices=batch.source_indices,
            metadata={
                "backend": self.name,
                "checkpoint": str(self.checkpoint),
                "device": self.device_name,
                "batch_size": self.batch_size,
                "atom_count": batch.atom_count,
                "grid_count": batch.geometry_count,
                "species": list(batch.species),
                "cutoff": float(getattr(model.pbc_distance, "cutoff", np.nan)),
                "max_num_neighbors": int(getattr(model.pbc_distance, "max_num_neighbors", -1)),
                "elapsed_seconds": time.perf_counter() - start,
            },
        )

    @property
    def device(self) -> Any:
        torch, _ = self._load_model()
        return torch.device(self.device_name)

    def _load_model(self) -> tuple[Any, Any]:
        if self._model is not None and self._torch is not None:
            return self._torch, self._model
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"ViSNet checkpoint not found: {self.checkpoint}")
        self._ensure_visnet_importable()
        import torch
        from visnet.model import VisNetEIP

        checkpoint = _torch_load(torch, self.checkpoint, self.device_name)
        args = checkpoint.get("args", {})
        model = VisNetEIP(
            hidden_channels=int(_get_arg(args, "hidden_channels", 64)),
            num_layers=int(_get_arg(args, "num_layers", 6)),
            num_heads=int(_get_arg(args, "num_heads", 8)),
            num_rbf=int(_get_arg(args, "num_rbf", 32)),
            cutoff=float(_get_arg(args, "cutoff", 6.0)),
            max_num_neighbors=int(_get_arg(args, "max_num_neighbors", 64)),
            mean=_to_float(checkpoint.get("energy_mean", 0.0)),
            std=_to_float(checkpoint.get("energy_std", 1.0)),
        )
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state)
        model.to(torch.device(self.device_name))
        model.eval()
        self._torch = torch
        self._model = model
        return torch, model

    def _cell_tensor(self, torch: Any, cell: np.ndarray, geometry_count: int) -> Any:
        array = np.asarray(cell, dtype=np.float64)
        if array.shape == (3,):
            array = np.diag(array)
        if array.shape == (3, 3):
            array = np.repeat(array.reshape(1, 3, 3), geometry_count, axis=0)
        elif array.shape == (geometry_count, 3):
            array = np.stack([np.diag(row) for row in array], axis=0)
        elif array.shape != (geometry_count, 3, 3):
            raise ValueError(f"Unsupported cell shape for VisNet: {array.shape}")
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def _energy_only(self, model: Any, z: Any, pos: Any, batch: Any, cell: Any) -> Any:
        model.pbc_distance.set_graph(cell=cell, edge_index=None, cell_shift=None)
        x, v = model._representation(z, pos, batch)
        atom_energy = model.backbone.output_model.pre_reduce(x, v)
        atom_energy = atom_energy * model.backbone.std
        if model.backbone.prior_model is not None:
            atom_energy = model.backbone.prior_model(atom_energy, z)
        energy = model.scatter(atom_energy, batch, dim=0, reduce=model.backbone.reduce_op)
        return (energy + model.backbone.mean).view(-1)

    def _ensure_visnet_importable(self) -> None:
        current = Path(__file__).resolve()
        for parent in current.parents:
            candidate = parent / "NaOH_7.5M" / "visnet" / "model.py"
            if candidate.exists():
                project_dir = candidate.parent.parent
                if str(project_dir) not in sys.path:
                    sys.path.insert(0, str(project_dir))
                return
        raise ImportError("Could not locate NaOH_7.5M/visnet/model.py from the LQVE backend path")


def _torch_load(torch: Any, path: Path, device: str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _get_arg(args: Any, name: str, default: Any) -> Any:
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)


def _to_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)
