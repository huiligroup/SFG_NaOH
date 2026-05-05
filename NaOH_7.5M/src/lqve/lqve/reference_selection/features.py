"""ViSNet feature extraction for one probe molecule in one frame."""

from __future__ import annotations

from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np

from lqve.embedding.io import read_xyz
from lqve.qc.io import atomic_numbers, parse_cell

from .config import ReferenceSelectionConfig


def extract_query_feature(config: ReferenceSelectionConfig) -> dict[str, Any]:
    """Return the flattened probe descriptor and frame metadata."""

    config.validate()
    frame = _load_frame(config)
    species = frame["species"]
    probe_indices = _resolve_probe_indices(species, config.mol_id, config.probe_indices)
    model, torch, model_info = _load_model(Path(config.checkpoint), config.device)
    z = torch.as_tensor(frame["z"], dtype=torch.long, device=torch.device(config.device))
    pos = torch.as_tensor(frame["pos"], dtype=torch.float32, device=torch.device(config.device))
    batch = torch.zeros(z.numel(), dtype=torch.long, device=torch.device(config.device))
    cell = torch.as_tensor(frame["cell"][None, :, :], dtype=torch.float32, device=torch.device(config.device))
    with torch.no_grad():
        atom_features = model.extract_atom_features(z=z, pos=pos, batch=batch, cell=cell)["atom_features"]
    atom_features_np = atom_features.detach().cpu().numpy().astype(np.float32, copy=False)
    feature = atom_features_np[probe_indices].reshape(-1).astype(np.float32, copy=False)
    return {
        "feature": feature,
        "atom_features": atom_features_np,
        "probe_indices": probe_indices,
        "probe_species": [species[idx] for idx in probe_indices],
        "species": species,
        "frame_index": int(config.frame_index),
        "mol_id": None if config.mol_id is None else int(config.mol_id),
        "source": frame["source"],
        "traj": frame.get("traj"),
        "frame": frame.get("frame"),
        "cell": np.asarray(frame["cell"], dtype=np.float64),
        "model_info": model_info,
    }


def _load_frame(config: ReferenceSelectionConfig) -> dict[str, Any]:
    if config.feature_data is not None:
        with Path(config.feature_data).open("rb") as handle:
            data = pickle.load(handle)
        idx = int(config.frame_index)
        if idx >= len(data["pos"]):
            raise IndexError(f"frame_index={idx} outside feature_data length {len(data['pos'])}")
        species = [str(item) for item in np.asarray(data["species"], dtype=object).tolist()]
        return {
            "species": species,
            "z": np.asarray(data["z"], dtype=np.int64),
            "pos": np.asarray(data["pos"][idx], dtype=np.float32),
            "cell": _normalize_cell(np.asarray(data["cell"][idx], dtype=np.float64)),
            "source": str(config.feature_data),
            "traj": str(data["traj"][idx]) if "traj" in data else None,
            "frame": int(data["frame"][idx]) if "frame" in data else None,
        }

    species, positions, _ = read_xyz(config.system_xyz, frame_index=config.frame_index)
    if config.cell is None:
        raise ValueError("--cell is required for reference selection from system_xyz")
    return {
        "species": species,
        "z": atomic_numbers(species),
        "pos": np.asarray(positions, dtype=np.float32),
        "cell": _normalize_cell(parse_cell(config.cell)),
        "source": str(config.system_xyz),
        "traj": None,
        "frame": config.frame_index,
    }


def _resolve_probe_indices(
    species: list[str],
    mol_id: int | None,
    probe_indices: list[int] | None,
) -> list[int]:
    if probe_indices is not None:
        if max(probe_indices) >= len(species):
            raise ValueError("probe_indices exceed frame atom count")
        return list(probe_indices)
    if mol_id is None:
        raise ValueError("mol_id or probe_indices is required")
    molecules = _build_default_na12_molecules(species)
    if mol_id < 0 or mol_id >= len(molecules):
        raise ValueError(f"mol_id out of range 0-{len(molecules) - 1}: {mol_id}")
    return list(molecules[mol_id]["atom_indices"])


def _build_default_na12_molecules(species: list[str]) -> list[dict[str, Any]]:
    if len(species) != 480:
        raise ValueError("mol_id lookup currently expects the Na12 480-atom ordering")
    molecules: list[dict[str, Any]] = []
    for mol_id in range(148):
        atom_indices = [3 * mol_id, 3 * mol_id + 1, 3 * mol_id + 2]
        mol_species = [species[idx] for idx in atom_indices]
        if mol_species != ["O", "H", "H"]:
            raise ValueError(f"Water molecule {mol_id} does not match O,H,H: {mol_species}")
        molecules.append({"mol_id": mol_id, "kind": "water", "atom_indices": atom_indices})
    offset = 148 * 3
    for local_id in range(12):
        mol_id = 148 + local_id
        atom_indices = [offset + 3 * local_id, offset + 3 * local_id + 1, offset + 3 * local_id + 2]
        mol_species = [species[idx] for idx in atom_indices]
        if mol_species != ["Na", "O", "H"]:
            raise ValueError(f"NaOH molecule {mol_id} does not match Na,O,H: {mol_species}")
        molecules.append({"mol_id": mol_id, "kind": "naoh", "atom_indices": atom_indices})
    return molecules


def _load_model(checkpoint_path: Path, device: str) -> tuple[Any, Any, dict[str, Any]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"ViSNet checkpoint not found: {checkpoint_path}")
    _ensure_visnet_importable()
    import torch
    from visnet.model import VisNetEIP

    checkpoint = _torch_load(torch, checkpoint_path, device)
    args = checkpoint.get("args", {})
    model_info = {
        "hidden_channels": int(_get_arg(args, "hidden_channels", 64)),
        "num_layers": int(_get_arg(args, "num_layers", 6)),
        "num_heads": int(_get_arg(args, "num_heads", 8)),
        "num_rbf": int(_get_arg(args, "num_rbf", 32)),
        "cutoff": float(_get_arg(args, "cutoff", 6.0)),
        "max_num_neighbors": int(_get_arg(args, "max_num_neighbors", 64)),
        "energy_mean": _to_float(checkpoint.get("energy_mean", 0.0)),
        "energy_std": _to_float(checkpoint.get("energy_std", 1.0)),
        "epoch": checkpoint.get("epoch"),
        "train_step": checkpoint.get("train_step"),
        "best_metric": checkpoint.get("best_metric"),
    }
    model = VisNetEIP(
        hidden_channels=model_info["hidden_channels"],
        num_layers=model_info["num_layers"],
        num_heads=model_info["num_heads"],
        num_rbf=model_info["num_rbf"],
        cutoff=model_info["cutoff"],
        max_num_neighbors=model_info["max_num_neighbors"],
        mean=model_info["energy_mean"],
        std=model_info["energy_std"],
    )
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.to(torch.device(device))
    model.eval()
    return model, torch, model_info


def _ensure_visnet_importable() -> None:
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "NaOH_7.5M" / "visnet" / "model.py"
        if candidate.exists():
            project_dir = candidate.parent.parent
            if str(project_dir) not in sys.path:
                sys.path.insert(0, str(project_dir))
            return
    raise ImportError("Could not locate NaOH_7.5M/visnet/model.py from LQVE reference selection")


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


def _normalize_cell(cell: np.ndarray) -> np.ndarray:
    array = np.asarray(cell, dtype=np.float64)
    if array.shape == (3,):
        return np.diag(array)
    if array.shape == (3, 3):
        return array
    raise ValueError(f"Unsupported cell shape for reference selection: {array.shape}")
