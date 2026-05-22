"""Checkpoint helpers shared by training, inference, and active learning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .model import VisNetEIP


def torch_load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint_path = Path(path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict at {checkpoint_path}, got {type(checkpoint)!r}")
    return checkpoint


def checkpoint_args(checkpoint: dict[str, Any]) -> dict[str, Any]:
    raw = checkpoint.get("args", {})
    if isinstance(raw, dict):
        return dict(raw)
    return {name: getattr(raw, name) for name in dir(raw) if not name.startswith("_")}


def get_checkpoint_arg(checkpoint: dict[str, Any], name: str, default: Any) -> Any:
    args = checkpoint.get("args", {})
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)


def checkpoint_model_kwargs(
    checkpoint: dict[str, Any],
    *,
    mean: float | None = None,
    std: float | None = None,
) -> dict[str, Any]:
    return {
        "hidden_channels": int(get_checkpoint_arg(checkpoint, "hidden_channels", 64)),
        "num_layers": int(get_checkpoint_arg(checkpoint, "num_layers", 6)),
        "num_heads": int(get_checkpoint_arg(checkpoint, "num_heads", 8)),
        "num_rbf": int(get_checkpoint_arg(checkpoint, "num_rbf", 32)),
        "cutoff": float(get_checkpoint_arg(checkpoint, "cutoff", 6.0)),
        "max_num_neighbors": int(get_checkpoint_arg(checkpoint, "max_num_neighbors", 64)),
        "mean": float(checkpoint.get("energy_mean", 0.0) if mean is None else mean),
        "std": float(checkpoint.get("energy_std", 1.0) if std is None else std),
    }


def checkpoint_model_info(checkpoint: dict[str, Any]) -> dict[str, Any]:
    info = checkpoint_model_kwargs(checkpoint)
    info["energy_mean"] = float(info.pop("mean"))
    info["energy_std"] = float(info.pop("std"))
    info["epoch"] = checkpoint.get("epoch")
    info["train_step"] = checkpoint.get("train_step")
    info["best_metric"] = checkpoint.get("best_metric")
    return info


def build_model_from_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    mean: float | None = None,
    std: float | None = None,
    eval_mode: bool = True,
) -> tuple[VisNetEIP, dict[str, Any], dict[str, Any]]:
    checkpoint = torch_load_checkpoint(path, device=device)
    model = VisNetEIP(**checkpoint_model_kwargs(checkpoint, mean=mean, std=std))
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.to(torch.device(device))
    if eval_mode:
        model.eval()
    return model, checkpoint, checkpoint_model_info(checkpoint)
