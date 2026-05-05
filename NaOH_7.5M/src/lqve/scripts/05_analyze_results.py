#!/usr/bin/env python
"""Analyze LQVE frequency-shift outputs and generate tables/figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = THIS_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lqve.analysis import AnalysisConfig, analyze_lqve_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize LQVE frequencies and generate analysis products."
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config.")
    parser.add_argument(
        "--shifts",
        type=Path,
        default=None,
        help="Directory containing shifts_summary.csv, or a shifts_summary.csv file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for final tables, figures, and logs.",
    )
    parser.add_argument(
        "--transitions",
        default=None,
        help="Optional one-based comma-separated transitions to analyze, e.g. 1,3,5.",
    )
    parser.add_argument(
        "--frame-axis",
        default=None,
        choices=("index", "frame_id"),
        help="X axis for time series plots.",
    )
    parser.add_argument(
        "--plot-format",
        default=None,
        choices=("png", "pdf", "svg"),
        help="Figure output format.",
    )
    parser.add_argument("--report-title", default=None, help="Title used in report.md.")
    parser.add_argument("--run-name", default=None, help="Optional subdirectory name under --output-dir.")
    parser.add_argument("--dvr-data", type=Path, default=None, help="Optional DVR data for diagnostics.")
    parser.add_argument(
        "--include-dvr-diagnostics",
        action="store_true",
        help="Generate optional DVR wavefunction diagnostic figure.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> AnalysisConfig:
    if args.config is not None:
        with args.config.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        overrides = {
            "shifts": args.shifts,
            "output_dir": args.output_dir,
            "transitions": parse_transitions(args.transitions),
            "frame_axis": args.frame_axis,
            "plot_format": args.plot_format,
            "report_title": args.report_title,
            "run_name": args.run_name,
            "dvr_data": args.dvr_data,
        }
        for key, value in overrides.items():
            if value is not None:
                raw[key] = str(value) if isinstance(value, Path) else value
        if args.include_dvr_diagnostics:
            raw["include_dvr_diagnostics"] = True
        return AnalysisConfig.from_dict(raw)

    return AnalysisConfig(
        shifts=args.shifts or Path("NaOH_7.5M/src/lqve/data/shifts"),
        output_dir=args.output_dir or Path("NaOH_7.5M/src/lqve/results"),
        transitions=parse_transitions(args.transitions),
        frame_axis=args.frame_axis or "index",
        plot_format=args.plot_format or "png",
        report_title=args.report_title or "LQVE Analysis Report",
        run_name=args.run_name,
        dvr_data=args.dvr_data,
        include_dvr_diagnostics=args.include_dvr_diagnostics,
    )


def parse_transitions(value: str | None) -> list[int] | None:
    if value is None:
        return None
    transitions = [int(item.strip()) for item in value.split(",") if item.strip()]
    return transitions or None


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    output_dir = config.resolved_output_dir()
    print(
        "analyzing_lqve_results "
        f"shifts={config.shifts} output_dir={output_dir} "
        f"transitions={config.transitions or 'all'} plot_format={config.plot_format}"
    )
    try:
        result = analyze_lqve_results(config)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"analysis_error: {exc}") from exc
    print(
        "finished_lqve_analysis "
        f"tables={len(result.tables)} figures={len(result.figures)} report={result.report}"
    )


if __name__ == "__main__":
    main()
