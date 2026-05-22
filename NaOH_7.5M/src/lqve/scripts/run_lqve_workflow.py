#!/usr/bin/env python
"""Run the end-to-end LQVE workflow without the analysis stage."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = THIS_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lqve.workflow import WorkflowConfig, run_lqve_workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DVR -> embedding -> energy -> shift and write one frequency CSV.")
    parser.add_argument("--config", type=Path, required=True, help="End-to-end workflow TOML or JSON config.")
    parser.add_argument("--run-name", default=None, help="Override run_name from config.")
    parser.add_argument("--limit-frames", type=int, default=None, help="Override limit_frames for smoke testing.")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume and recompute missing steps.")
    parser.add_argument("--overwrite", action="store_true", help="Remove run outputs that this workflow owns before recomputing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = WorkflowConfig.from_file(args.config).to_dict()
    if args.run_name is not None:
        raw["run_name"] = args.run_name
    if args.limit_frames is not None:
        raw["limit_frames"] = args.limit_frames
    if args.no_resume:
        raw["resume"] = False
    if args.overwrite:
        raw["overwrite"] = True
        raw["resume"] = False
    config = WorkflowConfig.from_dict(raw)
    print(
        "running_lqve_workflow "
        f"run_name={config.run_name} references={len(config.references)} "
        f"trajectories={len(config.trajectories)} dynamic_reference={config.dynamic_reference}"
    )
    result = run_lqve_workflow(config)
    print(
        "finished_lqve_workflow "
        f"frames={result.frame_count} all_frequencies={result.all_frequencies_csv} "
        f"reference_library={result.reference_library}"
    )


if __name__ == "__main__":
    main()
