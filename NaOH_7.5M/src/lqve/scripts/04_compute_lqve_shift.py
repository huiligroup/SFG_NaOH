#!/usr/bin/env python
"""Compute LQVE instantaneous frequencies and frequency shifts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = THIS_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lqve.perturbation import ShiftConfig, run_shift


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute instantaneous levels, transitions, and LQVE frequency shifts."
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config.")
    parser.add_argument(
        "--dvr-data",
        type=Path,
        default=None,
        help="Reference DVR result directory containing dvr_result.npz.",
    )
    parser.add_argument(
        "--qc-outputs",
        type=Path,
        default=None,
        help="QC output directory, frame directory, or qc_energies.npz.",
    )
    parser.add_argument(
        "--reference-energies",
        type=Path,
        default=None,
        help="Reference grid energies in Hartree. Supports npy, npz, csv, txt, dat.",
    )
    parser.add_argument(
        "--reference-key",
        default=None,
        help="Optional NPZ key or named text/CSV column for reference energies.",
    )
    parser.add_argument(
        "--reference-mode",
        default=None,
        choices=("file", "first-qc"),
        help="Use explicit reference file or the first QC frame as a smoke-test reference.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for frequency-shift data.",
    )
    parser.add_argument("--run-name", default=None, help="Optional subdirectory name under --output-dir.")
    parser.add_argument("--n-contract", type=int, default=None, help="Number of DVR states in effective Hamiltonian.")
    parser.add_argument("--n-transitions", type=int, default=None, help="Number of transitions to output.")
    parser.add_argument("--limit-frames", type=int, default=None, help="Optional number of QC frames to process.")
    parser.add_argument(
        "--keep-mean",
        action="store_true",
        help="Do not subtract the mean perturbation energy before diagonalization.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ShiftConfig:
    if args.config is not None:
        with args.config.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        overrides = {
            "dvr_data": args.dvr_data,
            "qc_outputs": args.qc_outputs,
            "reference_energies": args.reference_energies,
            "reference_key": args.reference_key,
            "reference_mode": args.reference_mode,
            "output_dir": args.output_dir,
            "run_name": args.run_name,
            "n_contract": args.n_contract,
            "n_transitions": args.n_transitions,
            "limit_frames": args.limit_frames,
        }
        for key, value in overrides.items():
            if value is not None:
                raw[key] = str(value) if isinstance(value, Path) else value
        if args.keep_mean:
            raw["subtract_mean"] = False
        return ShiftConfig.from_dict(raw)

    return ShiftConfig(
        dvr_data=args.dvr_data,
        qc_outputs=args.qc_outputs or Path("NaOH_7.5M/src/lqve/data/qc_outputs"),
        reference_energies=args.reference_energies,
        reference_key=args.reference_key,
        reference_mode=args.reference_mode or "file",
        output_dir=args.output_dir or Path("NaOH_7.5M/src/lqve/data/shifts"),
        run_name=args.run_name,
        n_contract=args.n_contract or 30,
        n_transitions=args.n_transitions or 6,
        limit_frames=args.limit_frames,
        subtract_mean=not args.keep_mean,
    )


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    output_dir = config.resolved_output_dir()
    print(
        "computing_lqve_shift "
        f"dvr_data={config.dvr_data} qc_outputs={config.qc_outputs} "
        f"reference_mode={config.reference_mode} output_dir={output_dir} "
        f"n_contract={config.n_contract} n_transitions={config.n_transitions}"
    )
    results = run_shift(config)
    success_count = sum(result.success for result in results)
    print(
        "finished_lqve_shift "
        f"frames={len(results)} success={success_count} "
        f"summary={output_dir / 'shifts_summary.csv'}"
    )


if __name__ == "__main__":
    main()
