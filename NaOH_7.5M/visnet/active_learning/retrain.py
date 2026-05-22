"""Warm-start retraining orchestration for active learning rounds."""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

from visnet.checkpoint import checkpoint_model_info, torch_load_checkpoint

from .config import RetrainConfig


ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = ROOT / "visnet" / "train.py"


def run_retraining(
    config: RetrainConfig,
    *,
    dataset_path: str | Path,
    init_checkpoint: str | Path,
    output_dir: str | Path,
) -> Path:
    dataset_path = Path(dataset_path)
    init_checkpoint = Path(init_checkpoint)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch_load_checkpoint(init_checkpoint, device=config.device)
    info = checkpoint_model_info(checkpoint)
    edge_path = output_dir / f"edges_cutoff{info['cutoff']:.1f}.pt"
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--data",
        str(dataset_path),
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(config.epochs),
        "--frame-stride",
        str(config.frame_stride),
        "--batch-size",
        str(config.batch_size),
        "--grad-accumulation-steps",
        str(config.grad_accumulation_steps),
        "--lr",
        str(config.lr),
        "--device",
        config.device,
        "--hidden-channels",
        str(info["hidden_channels"]),
        "--num-layers",
        str(info["num_layers"]),
        "--num-heads",
        str(info["num_heads"]),
        "--num-rbf",
        str(info["num_rbf"]),
        "--cutoff",
        str(info["cutoff"]),
        "--max-num-neighbors",
        str(info["max_num_neighbors"]),
        "--grad-clip",
        str(config.grad_clip),
        "--batches-per-step",
        str(config.batches_per_step),
        "--val-interval-steps",
        str(config.val_interval_steps),
        "--early-stopping-patience",
        str(config.early_stopping_patience),
        "--early-stopping-min-delta",
        str(config.early_stopping_min_delta),
        "--init-checkpoint",
        str(init_checkpoint),
    ]
    if config.limit_batches is not None:
        command.extend(["--limit-batches", str(config.limit_batches)])
    if config.limit_val_batches is not None:
        command.extend(["--limit-val-batches", str(config.limit_val_batches)])
    if config.gradient_checkpointing:
        command.append("--gradient-checkpointing")
    if config.build_edges_if_missing:
        command.extend(["--edge-data", str(edge_path), "--build-edges-if-missing"])
    if config.wandb:
        command.extend(["--wandb", "--wandb-project", config.wandb_project])
        if config.wandb_name:
            command.extend(["--wandb-name", config.wandb_name])

    completed = subprocess.run(
        command,
        cwd=ROOT.parent,
        text=True,
        capture_output=True,
        check=False,
    )
    (output_dir / "train.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "train.stderr.log").write_text(completed.stderr, encoding="utf-8")
    (output_dir / "train.command.txt").write_text(" ".join(shlex.quote(part) for part in command) + "\n", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Retraining failed with return code {completed.returncode}; see {output_dir/'train.stderr.log'}")

    best_path = output_dir / "best.pt"
    smoke_path = output_dir / "visnet_eip_smoke.pt"
    if best_path.exists():
        return best_path
    if smoke_path.exists():
        return smoke_path
    raise FileNotFoundError(f"Retraining completed but no checkpoint found under {output_dir}")
