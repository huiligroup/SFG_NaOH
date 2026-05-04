"""Train or smoke-test ViSNet + evidential force uncertainty on Na12 data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from visnet.data.dataset import NaOHDataset, collate_frames
    from visnet.data.precompute_edges import build_edge_store, default_edge_path, selected_indices
    from visnet.losses import eip_loss
    from visnet.model import VisNetEIP
else:
    from .data.dataset import NaOHDataset, collate_frames
    from .data.precompute_edges import build_edge_store, default_edge_path, selected_indices
    from .losses import eip_loss
    from .model import VisNetEIP


def move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


class MetricAccumulator:
    def __init__(self, collect_uncertainty: bool = False) -> None:
        self.collect_uncertainty = collect_uncertainty
        self.loss_sum = 0.0
        self.energy_loss_sum = 0.0
        self.force_nll_sum = 0.0
        self.evidence_reg_sum = 0.0
        self.energy_abs_sum = 0.0
        self.energy_sq_sum = 0.0
        self.force_abs_sum = 0.0
        self.force_sq_sum = 0.0
        self.batch_count = 0
        self.energy_count = 0
        self.force_count = 0
        self.epistemic_values: list[torch.Tensor] = []
        self.aleatoric_values: list[torch.Tensor] = []
        self.frame_rows: list[dict[str, float | int | str]] = []

    def update(self, losses: dict[str, torch.Tensor], out: dict[str, torch.Tensor], batch: dict) -> None:
        energy_error = (out["energy"].view_as(batch["energy"]) - batch["energy"]).detach()
        force_error = (out["force"] - batch["forces"]).detach()
        self.loss_sum += float(losses["loss"].detach().cpu())
        self.energy_loss_sum += float(losses["energy_l1"].detach().cpu())
        self.force_nll_sum += float(losses["force_nll"].detach().cpu())
        self.evidence_reg_sum += float(losses["evidence_reg"].detach().cpu())
        self.energy_abs_sum += float(energy_error.abs().sum().cpu())
        self.energy_sq_sum += float(energy_error.square().sum().cpu())
        self.force_abs_sum += float(force_error.abs().sum().cpu())
        self.force_sq_sum += float(force_error.square().sum().cpu())
        self.batch_count += 1
        self.energy_count += int(energy_error.numel())
        self.force_count += int(force_error.numel())
        if self.collect_uncertainty:
            epistemic = out["epistemic"].detach()
            aleatoric = out["aleatoric"].detach()
            self.epistemic_values.append(epistemic.flatten().cpu())
            self.aleatoric_values.append(aleatoric.flatten().cpu())
            self._add_frame_rows(batch, energy_error, force_error, epistemic, aleatoric)

    def _add_frame_rows(
        self,
        batch: dict,
        energy_error: torch.Tensor,
        force_error: torch.Tensor,
        epistemic: torch.Tensor,
        aleatoric: torch.Tensor,
    ) -> None:
        graph_count = int(batch["energy"].numel())
        for graph_idx in range(graph_count):
            mask = batch["batch"] == graph_idx
            frame_force_error = force_error[mask].abs().flatten()
            frame_epistemic = epistemic[mask].flatten()
            frame_aleatoric = aleatoric[mask].flatten()
            self.frame_rows.append(
                {
                    "traj": str(batch["traj"][graph_idx]),
                    "frame": int(batch["frame"][graph_idx]),
                    "frame_index": int(batch["frame_index"][graph_idx]),
                    "energy_abs_error": float(energy_error[graph_idx].abs().cpu()),
                    "force_mae": float(frame_force_error.mean().cpu()),
                    "force_rmse": float(frame_force_error.square().mean().sqrt().cpu()),
                    "epistemic_mean": float(frame_epistemic.mean().cpu()),
                    "epistemic_p95": float(torch.quantile(frame_epistemic, 0.95).cpu()),
                    "epistemic_max": float(frame_epistemic.max().cpu()),
                    "aleatoric_mean": float(frame_aleatoric.mean().cpu()),
                    "aleatoric_p95": float(torch.quantile(frame_aleatoric, 0.95).cpu()),
                    "aleatoric_max": float(frame_aleatoric.max().cpu()),
                }
            )

    def compute(self) -> dict[str, float]:
        if self.batch_count == 0:
            raise ValueError("No batches were accumulated")
        metrics = {
            "loss": self.loss_sum / self.batch_count,
            "energy_loss": self.energy_loss_sum / self.batch_count,
            "force_nll": self.force_nll_sum / self.batch_count,
            "evidence_reg": self.evidence_reg_sum / self.batch_count,
            "energy_mae": self.energy_abs_sum / max(self.energy_count, 1),
            "energy_rmse": (self.energy_sq_sum / max(self.energy_count, 1)) ** 0.5,
            "force_mae": self.force_abs_sum / max(self.force_count, 1),
            "force_rmse": (self.force_sq_sum / max(self.force_count, 1)) ** 0.5,
            "batches": float(self.batch_count),
        }
        if self.collect_uncertainty and self.epistemic_values:
            epistemic = torch.cat(self.epistemic_values)
            aleatoric = torch.cat(self.aleatoric_values)
            metrics.update(
                {
                    "epistemic_mean": float(epistemic.mean()),
                    "epistemic_p95": float(torch.quantile(epistemic, 0.95)),
                    "epistemic_max": float(epistemic.max()),
                    "aleatoric_mean": float(aleatoric.mean()),
                    "aleatoric_p95": float(torch.quantile(aleatoric, 0.95)),
                    "aleatoric_max": float(aleatoric.max()),
                    "uncertainty_error_spearman": spearman_from_rows(self.frame_rows),
                }
            )
        return metrics


def spearman_from_rows(rows: list[dict[str, float | int | str]]) -> float:
    if len(rows) < 2:
        return 0.0
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return 0.0
    force_errors = [float(row["force_mae"]) for row in rows]
    uncertainties = [float(row["epistemic_p95"]) for row in rows]
    result = spearmanr(uncertainties, force_errors)
    if result.correlation != result.correlation:
        return 0.0
    return float(result.correlation)


def compute_losses(model: VisNetEIP, batch: dict, args: argparse.Namespace) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    out = model(
        batch["z"],
        batch["pos"],
        batch["batch"],
        batch["cell"],
        edge_index=batch.get("edge_index"),
        cell_shift=batch.get("cell_shift"),
    )
    losses = eip_loss(
        pred_energy=out["energy"],
        true_energy=batch["energy"],
        pred_force=out["force"],
        true_force=batch["forces"],
        nu=out["nu"],
        alpha=out["alpha"],
        beta=out["beta"],
        force_weight=args.force_weight,
        reg_weight=args.reg_weight,
        q=args.quantile,
    )
    if not torch.isfinite(losses["loss"]):
        raise FloatingPointError(f"Non-finite loss: {losses}")
    return losses, out


def run_validation(
    model: VisNetEIP,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, float], list[dict[str, float | int | str]]]:
    model.eval()
    metrics = MetricAccumulator(collect_uncertainty=True)
    limit_batches = args.limit_val_batches
    if limit_batches is None and args.limit_batches is not None:
        limit_batches = args.limit_batches
    for batch_idx, batch in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(True):
            losses, out = compute_losses(model, batch, args)
        metrics.update(losses, out, batch)
    return metrics.compute(), metrics.frame_rows


def print_epoch_summary(epoch: int, metrics: dict[str, float]) -> None:
    ordered = " ".join(f"{key}={value:.6g}" for key, value in sorted(metrics.items()))
    print(f"epoch={epoch} summary {ordered}")


def format_metrics(prefix: str, metrics: dict[str, float]) -> str:
    fields = (
        "loss",
        "energy_mae",
        "force_mae",
        "energy_rmse",
        "force_rmse",
        "epistemic_mean",
        "epistemic_p95",
        "epistemic_max",
        "aleatoric_mean",
        "uncertainty_error_spearman",
        "force_nll",
        "evidence_reg",
    )
    return " ".join(f"{prefix}/{key}={metrics[key]:.6g}" for key in fields if key in metrics)


def write_high_uncertainty_frames(
    output_dir: Path,
    rows: list[dict[str, float | int | str]],
    epoch: int,
    train_step: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "high_uncertainty_frames.csv"
    sorted_rows = sorted(rows, key=lambda row: float(row["epistemic_p95"]), reverse=True)
    fieldnames = [
        "rank",
        "epoch",
        "train_step",
        "traj",
        "frame",
        "frame_index",
        "force_mae",
        "force_rmse",
        "energy_abs_error",
        "epistemic_mean",
        "epistemic_p95",
        "epistemic_max",
        "aleatoric_mean",
        "aleatoric_p95",
        "aleatoric_max",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(sorted_rows, start=1):
            writer.writerow({"rank": rank, "epoch": epoch, "train_step": train_step} | row)
    return path


def save_checkpoint(
    path: Path,
    model: VisNetEIP,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    mean: float,
    std: float,
    history: list[dict[str, float | int | str]],
    epoch: int,
    train_step: int,
    best_metric: float | None,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "energy_mean": mean,
            "energy_std": std,
            "history": history,
            "epoch": epoch,
            "train_step": train_step,
            "best_metric": best_metric,
        },
        path,
    )


def init_wandb(args: argparse.Namespace, model: VisNetEIP) -> Any | None:
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb logging was requested, but wandb is not installed.") from exc

    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        mode=args.wandb_mode,
        config=config,
        dir=str(args.output_dir),
    )
    if args.wandb_watch:
        wandb.watch(model, log=args.wandb_watch_log, log_freq=args.batches_per_step)
    return run


def load_edge_store(args: argparse.Namespace) -> tuple[dict | None, Path | None]:
    edge_path = args.edge_data if args.edge_data is not None else default_edge_path(args.data, args.cutoff)
    if edge_path.exists():
        store = torch.load(edge_path, map_location="cpu", weights_only=False)
        print(
            f"loaded_edge_data={edge_path} frames={len(store['frame_indices'])} "
            f"avg_neighbors={store.get('stats', {}).get('average_neighbors', float('nan')):.3f}"
        )
        return store, edge_path
    if not args.build_edges_if_missing:
        print(f"edge_data_missing={edge_path}; using runtime no_grad neighbor construction")
        return None, edge_path
    store = build_edge_store(
        data_path=args.data,
        output_path=edge_path,
        cutoff=args.cutoff,
        max_num_neighbors=args.max_num_neighbors,
        frame_stride=args.frame_stride,
        splits={"train", "val", "test"},
        limit=args.edge_build_limit,
    )
    return store, edge_path


def expected_frame_indices(data_path: Path, frame_stride: int, limit: int | None, splits: set[str]) -> set[int]:
    with data_path.open("rb") as handle:
        import pickle

        dataset = pickle.load(handle)
    return set(map(int, selected_indices(dataset, splits=splits, frame_stride=frame_stride, limit=limit)))


def report_edge_coverage(edge_store: dict | None, args: argparse.Namespace) -> None:
    if edge_store is None:
        return
    required_splits = {"train"} if args.no_val else {"train", "val"}
    required = expected_frame_indices(args.data, args.frame_stride, args.limit_frames, required_splits)
    available = set(map(int, edge_store["frame_indices"]))
    missing = sorted(required.difference(available))
    extra = sorted(available.difference(required))
    if missing:
        print(
            f"edge_data_warning missing_required_frames={len(missing)} first_missing={missing[:10]} "
            "fallback_runtime_edges=enabled"
        )
    else:
        print(f"edge_data_coverage required_frames={len(required)} missing_required_frames=0")
    if edge_store.get("selection_mode") != "per_split":
        print(
            "edge_data_warning selection_mode is not per_split; regenerate with the current "
            "NaOH_7.5M/visnet/data/precompute_edges.py to avoid runtime fallback."
        )
    if extra:
        print(f"edge_data_note extra_frames={len(extra)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "visnet" / "na12_ab.pkl",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "runs" / "smoke")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--force-weight", type=float, default=1.0)
    parser.add_argument("--reg-weight", type=float, default=1.0e-2)
    parser.add_argument("--quantile", type=float, default=0.5)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-rbf", type=int, default=32)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--max-num-neighbors", type=int, default=64)
    parser.add_argument("--edge-data", type=Path, default=None)
    parser.add_argument("--build-edges-if-missing", action="store_true")
    parser.add_argument("--edge-build-limit", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batches-per-step", type=int, default=100)
    parser.add_argument("--val-interval-steps", type=int, default=10)
    parser.add_argument("--no-val", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--early-stopping-monitor", default="val/force_mae")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default="naoh-visnet-eip")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-mode", default="online", choices=("online", "offline", "disabled"))
    parser.add_argument("--wandb-watch", action="store_true", help="Log gradients/parameters with wandb.watch.")
    parser.add_argument("--wandb-watch-log", default="gradients", choices=("gradients", "parameters", "all"))
    args = parser.parse_args()
    if args.batches_per_step < 1:
        raise ValueError("--batches-per-step must be >= 1")
    if args.val_interval_steps < 1:
        raise ValueError("--val-interval-steps must be >= 1")

    edge_store, edge_path = load_edge_store(args)
    report_edge_coverage(edge_store, args)
    train_dataset = NaOHDataset(
        args.data,
        split="train",
        limit=args.limit_frames,
        frame_stride=args.frame_stride,
        edge_store=edge_store,
        cutoff=args.cutoff,
        max_num_neighbors=args.max_num_neighbors,
    )
    val_dataset = None if args.no_val else NaOHDataset(
        args.data,
        split="val",
        limit=args.limit_frames,
        frame_stride=args.frame_stride,
        edge_store=edge_store,
        cutoff=args.cutoff,
        max_num_neighbors=args.max_num_neighbors,
    )
    mean, std = train_dataset.energy_mean_std
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_frames,
        num_workers=0,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_frames,
            num_workers=0,
        )
    device = torch.device(args.device)
    model = VisNetEIP(
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_rbf=args.num_rbf,
        cutoff=args.cutoff,
        max_num_neighbors=args.max_num_neighbors,
        mean=mean,
        std=std,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = init_wandb(args, model)
    history: list[dict[str, float | int | str]] = []
    train_step = 0
    best_metric: float | None = None
    bad_validation_count = 0
    stop_training = False
    print(
        f"dataset train_frames={len(train_dataset)} "
        f"val_frames={0 if val_dataset is None else len(val_dataset)} frame_stride={args.frame_stride} "
        f"edge_data={'runtime' if edge_store is None else edge_path}"
    )
    try:
        for epoch in range(args.epochs):
            model.train()
            epoch_metrics = MetricAccumulator()
            step_metrics = MetricAccumulator()
            for batch_idx, batch in enumerate(train_loader):
                if args.limit_batches is not None and batch_idx >= args.limit_batches:
                    break
                batch = move_batch(batch, device)
                losses, out = compute_losses(model, batch, args)
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                epoch_metrics.update(losses, out, batch)
                step_metrics.update(losses, out, batch)

                is_step_boundary = step_metrics.batch_count >= args.batches_per_step
                is_epoch_last = batch_idx + 1 == len(train_loader)
                is_limited_last = args.limit_batches is not None and batch_idx + 1 >= args.limit_batches
                if is_step_boundary or is_epoch_last or is_limited_last:
                    train_step += 1
                    train_metrics = step_metrics.compute()
                    record: dict[str, float | int | str] = {
                        "epoch": epoch,
                        "train_step": train_step,
                        "batch": batch_idx,
                    } | {f"train/{key}": value for key, value in train_metrics.items()}
                    print(
                        f"epoch={epoch} step={train_step} batch={batch_idx} "
                        f"{format_metrics('train', train_metrics)}"
                    )

                    if val_loader is not None and train_step % args.val_interval_steps == 0:
                        val_metrics, val_frame_rows = run_validation(model, val_loader, device, args)
                        model.train()
                        record.update({f"val/{key}": value for key, value in val_metrics.items()})
                        print(f"epoch={epoch} step={train_step} validation {format_metrics('val', val_metrics)}")
                        high_uncertainty_path = write_high_uncertainty_frames(
                            args.output_dir,
                            val_frame_rows,
                            epoch=epoch,
                            train_step=train_step,
                        )
                        print(f"epoch={epoch} step={train_step} high_uncertainty_frames={high_uncertainty_path}")

                        monitor_value = record.get(args.early_stopping_monitor)
                        if monitor_value is None:
                            raise KeyError(f"Early stopping monitor not found: {args.early_stopping_monitor}")
                        improved = best_metric is None or float(monitor_value) < best_metric - args.early_stopping_min_delta
                        if improved:
                            best_metric = float(monitor_value)
                            bad_validation_count = 0
                            save_checkpoint(
                                args.output_dir / "best.pt",
                                model,
                                optimizer,
                                args,
                                mean,
                                std,
                                history,
                                epoch,
                                train_step,
                                best_metric,
                            )
                            print(
                                f"epoch={epoch} step={train_step} best_checkpoint "
                                f"{args.early_stopping_monitor}={best_metric:.6g}"
                            )
                        else:
                            bad_validation_count += 1
                            print(
                                f"epoch={epoch} step={train_step} early_stopping_wait="
                                f"{bad_validation_count}/{args.early_stopping_patience} "
                                f"best_{args.early_stopping_monitor}={best_metric:.6g}"
                            )
                            if bad_validation_count >= args.early_stopping_patience:
                                stop_training = True
                                record["stop_reason"] = "early_stopping"

                    history.append(record)
                    if wandb_run is not None:
                        wandb_run.log(record, step=train_step)
                    step_metrics = MetricAccumulator()
                    if stop_training:
                        break

            epoch_record = {"epoch": epoch, "train_step": train_step} | {
                f"epoch/train/{key}": value for key, value in epoch_metrics.compute().items()
            }
            print_epoch_summary(epoch, epoch_record)
            history.append(epoch_record)
            if wandb_run is not None:
                wandb_run.log(epoch_record, step=train_step)
            if stop_training:
                print(f"stopped_early epoch={epoch} train_step={train_step}")
                break
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    ckpt_path = args.output_dir / "visnet_eip_smoke.pt"
    save_checkpoint(ckpt_path, model, optimizer, args, mean, std, history, epoch, train_step, best_metric)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    print(f"saved_checkpoint={ckpt_path}")


if __name__ == "__main__":
    main()
