#!/usr/bin/env python
"""Evaluate energies for embedded DVR geometries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = THIS_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lqve.qc import QCConfig, run_qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run energy calculations for DVR geometries with QC or ViSNet backends."
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config.")
    parser.add_argument(
        "--geometries",
        type=Path,
        default=None,
        help="Embedding result directory, embedded_geometries.npz, or parent containing frame dirs.",
    )
    parser.add_argument(
        "--backend",
        default=None,
        choices=("xtb", "gaussian", "cp2k", "visnet", "mock"),
        help="Energy backend.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for structured energy outputs.",
    )
    parser.add_argument("--workers", type=int, default=None, help="Parallel workers for external backends.")
    parser.add_argument("--batch-size", type=int, default=None, help="Geometries per backend chunk.")
    parser.add_argument("--resume", action="store_true", help="Skip frames with existing qc_energies.npz.")
    parser.add_argument("--dry-run", action="store_true", help="Write a manifest plan without running energies.")
    parser.add_argument("--keep-workdirs", action="store_true", help="Keep backend work directories.")
    parser.add_argument("--limit-frames", type=int, default=None, help="Optional number of frame batches to run.")
    parser.add_argument(
        "--limit-geometries",
        type=int,
        default=None,
        help="Optional number of DVR grid geometries per frame.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="ViSNet checkpoint for --backend visnet.")
    parser.add_argument("--device", default=None, help="ViSNet device, e.g. cpu or cuda.")
    parser.add_argument(
        "--cell",
        default=None,
        help="Periodic cell as Lx,Ly,Lz or 9 comma-separated matrix values in Angstrom.",
    )
    parser.add_argument("--command", default=None, help="Backend command override.")
    parser.add_argument("--template", type=Path, default=None, help="Gaussian/CP2K input template.")
    parser.add_argument("--nproc", type=int, default=None, help="MPI process count for CP2K.")
    parser.add_argument("--method", default=None, help="xTB method flags when --command is not set.")
    parser.add_argument("--work-dir", type=Path, default=None, help="Directory for external backend work files.")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> QCConfig:
    if args.config is not None:
        with args.config.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        overrides = {
            "geometries": args.geometries,
            "backend": args.backend,
            "output_dir": args.output_dir,
            "workers": args.workers,
            "batch_size": args.batch_size,
            "checkpoint": args.checkpoint,
            "device": args.device,
            "cell": args.cell,
            "command": args.command,
            "template": args.template,
            "nproc": args.nproc,
            "method": args.method,
            "work_dir": args.work_dir,
            "limit_frames": args.limit_frames,
            "limit_geometries": args.limit_geometries,
        }
        for key, value in overrides.items():
            if value is not None:
                raw[key] = str(value) if isinstance(value, Path) else value
        if args.resume:
            raw["resume"] = True
        if args.dry_run:
            raw["dry_run"] = True
        if args.keep_workdirs:
            raw["keep_workdirs"] = True
        return QCConfig.from_dict(raw)

    return QCConfig(
        geometries=args.geometries or Path("NaOH_7.5M/src/lqve/data/embedded_geometries"),
        output_dir=args.output_dir or Path("NaOH_7.5M/src/lqve/data/qc_outputs"),
        backend=args.backend or "visnet",
        workers=args.workers or 1,
        batch_size=args.batch_size or 32,
        resume=args.resume,
        dry_run=args.dry_run,
        keep_workdirs=args.keep_workdirs,
        limit_frames=args.limit_frames,
        limit_geometries=args.limit_geometries,
        command=args.command,
        template=args.template,
        nproc=args.nproc or 1,
        method=args.method or "--gfn 2",
        checkpoint=args.checkpoint,
        device=args.device or "cpu",
        cell=_parse_cell_arg(args.cell),
        work_dir=args.work_dir,
    )


def _parse_cell_arg(value: str | None) -> list[float] | None:
    if value is None:
        return None
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    print(
        "running_qc_energies "
        f"backend={config.backend} geometries={config.geometries} output_dir={config.output_dir} "
        f"workers={config.workers} batch_size={config.batch_size} dry_run={config.dry_run}"
    )
    results = run_qc(config)
    geometry_count = sum(int(result.energies_hartree.size) for result in results)
    success_count = sum(int(result.success_mask.sum()) for result in results)
    print(
        "finished_qc_energies "
        f"frames={len(results)} geometries={geometry_count} success={success_count} "
        f"output_dir={config.output_dir}"
    )


if __name__ == "__main__":
    main()
