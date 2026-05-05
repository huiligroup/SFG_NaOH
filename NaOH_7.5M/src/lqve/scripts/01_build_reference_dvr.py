#!/usr/bin/env python
"""Build reference DVR data from a high-accuracy reference PES."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = THIS_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lqve.dvr import DVRConfig, ModeConfig, build_reference_dvr


def normalize_argv(argv: list[str]) -> list[str]:
    """Allow negative coordinate ranges after ``--ranges``.

    ``argparse`` treats values such as ``-0.7:0.7,-0.9:0.9`` as option-like
    tokens when they are separated from ``--ranges`` by a space. The documented
    command uses that form, so normalize it to ``--ranges=<value>`` first.
    """

    result: list[str] = []
    idx = 0
    while idx < len(argv):
        if argv[idx] == "--ranges" and idx + 1 < len(argv):
            result.append(f"--ranges={argv[idx + 1]}")
            idx += 2
        else:
            result.append(argv[idx])
            idx += 1
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reference vibrational levels, wavefunctions, and DVR points."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON workflow config. CLI values override only --pes and --output-dir.",
    )
    parser.add_argument("--pes", type=Path, default=None, help="Reference PES table.")
    parser.add_argument(
        "--reference-pes",
        type=Path,
        default=None,
        help="Alias for --pes kept for readability.",
    )
    parser.add_argument("--dims", type=int, default=None, help="Number of DVR dimensions.")
    parser.add_argument(
        "--mode-names",
        default=None,
        help="Comma-separated mode names, for example q1,q2,q3.",
    )
    parser.add_argument(
        "--ranges",
        default=None,
        help="Comma-separated coordinate ranges, for example -0.7:0.7,-0.9:0.9.",
    )
    parser.add_argument(
        "--sinc-points",
        default=None,
        help="One integer or comma-separated per-mode SINC-DVR point counts.",
    )
    parser.add_argument(
        "--podvr-points",
        default=None,
        help="One integer or comma-separated per-mode PODVR point counts.",
    )
    parser.add_argument(
        "--masses",
        default=None,
        help="Optional one value or comma-separated per-mode masses in electron masses.",
    )
    parser.add_argument("--states", type=int, default=20, help="Number of lowest states to solve.")
    parser.add_argument(
        "--coord-unit",
        default="angstrom",
        choices=("angstrom", "bohr"),
        help="Coordinate unit used by --ranges and PES coordinate columns.",
    )
    parser.add_argument(
        "--energy-unit",
        default="cm-1",
        choices=("cm-1", "hartree", "au", "ev"),
        help="Energy unit used by the PES energy column.",
    )
    parser.add_argument(
        "--dense-threshold",
        type=int,
        default=512,
        help="Use dense eigh when PODVR basis size is at or below this threshold.",
    )
    parser.add_argument(
        "--solver-tol",
        type=float,
        default=1.0e-10,
        help="Tolerance for sparse eigsh.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated DVR data.",
    )
    return parser.parse_args(normalize_argv(sys.argv[1:]))


def config_from_args(args: argparse.Namespace) -> tuple[Path, DVRConfig, Path]:
    if args.config is not None:
        with args.config.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        pes_path = Path(args.pes or args.reference_pes or raw.get("pes", ""))
        if not str(pes_path):
            raise ValueError("Config mode requires a PES path in JSON or --pes")
        output_dir = Path(
            args.output_dir
            or raw.get("output_dir")
            or default_output_dir(raw.get("run_name"), raw.get("modes", []))
        )
        config = DVRConfig.from_dict(raw)
        return pes_path, config, output_dir

    pes_path = args.pes or args.reference_pes
    if pes_path is None:
        raise ValueError("--pes is required when --config is not used")
    if args.dims is None:
        raise ValueError("--dims is required when --config is not used")
    dims = args.dims
    mode_names = parse_mode_names(args.mode_names, dims)
    ranges = parse_ranges(args.ranges, dims)
    sinc_points = parse_int_list(args.sinc_points, dims, "sinc-points")
    podvr_points = parse_int_list(args.podvr_points, dims, "podvr-points")
    masses = parse_float_list(args.masses, dims, "masses", default=1822.887427)
    modes = [
        ModeConfig(
            name=mode_names[idx],
            lower=ranges[idx][0],
            upper=ranges[idx][1],
            sinc_points=sinc_points[idx],
            podvr_points=podvr_points[idx],
            mass=masses[idx],
        )
        for idx in range(dims)
    ]
    config = DVRConfig(
        modes=modes,
        states=args.states,
        coord_unit=args.coord_unit,
        energy_unit=args.energy_unit,
        dense_threshold=args.dense_threshold,
        solver_tol=args.solver_tol,
    )
    config.validate()
    output_dir = args.output_dir or default_output_dir(None, [mode.__dict__ for mode in modes])
    return Path(pes_path), config, Path(output_dir)


def default_output_dir(run_name: str | None, modes: list[dict]) -> Path:
    if run_name:
        name = run_name
    elif modes:
        points = "_".join(str(mode.get("podvr_points", "x")) for mode in modes)
        name = f"dvr_{len(modes)}d_{points}"
    else:
        name = "dvr_run"
    return Path("NaOH_7.5M/src/lqve/data/dvr") / name


def parse_mode_names(value: str | None, dims: int) -> list[str]:
    if value is None:
        return [f"q{idx + 1}" for idx in range(dims)]
    names = [item.strip() for item in value.split(",") if item.strip()]
    if len(names) != dims:
        raise ValueError(f"--mode-names must contain {dims} values")
    return names


def parse_ranges(value: str | None, dims: int) -> list[tuple[float, float]]:
    if value is None:
        raise ValueError("--ranges is required when --config is not used")
    ranges: list[tuple[float, float]] = []
    for item in value.split(","):
        if ":" not in item:
            raise ValueError(f"Invalid range {item!r}; expected lower:upper")
        lower, upper = item.split(":", 1)
        ranges.append((float(lower), float(upper)))
    if len(ranges) != dims:
        raise ValueError(f"--ranges must contain {dims} ranges")
    return ranges


def parse_int_list(value: str | None, dims: int, label: str) -> list[int]:
    if value is None:
        raise ValueError(f"--{label} is required when --config is not used")
    values = [int(item) for item in value.split(",")]
    if len(values) == 1:
        values = values * dims
    if len(values) != dims:
        raise ValueError(f"--{label} must contain one value or {dims} values")
    return values


def parse_float_list(
    value: str | None,
    dims: int,
    label: str,
    default: float,
) -> list[float]:
    if value is None:
        return [default] * dims
    values = [float(item) for item in value.split(",")]
    if len(values) == 1:
        values = values * dims
    if len(values) != dims:
        raise ValueError(f"--{label} must contain one value or {dims} values")
    return values


def main() -> None:
    args = parse_args()
    pes_path, config, output_dir = config_from_args(args)
    print(
        "building_reference_dvr "
        f"pes={pes_path} dims={config.dims} basis_shape={config.basis_shape} "
        f"states={config.states} output_dir={output_dir}"
    )
    result = build_reference_dvr(pes_path, config)
    result.save(output_dir)
    print(
        "saved_reference_dvr "
        f"levels={output_dir / 'dvr_result.npz'} metadata={output_dir / 'metadata.json'} "
        f"lowest_level_hartree={result.levels_hartree[0]:.12g}"
    )


if __name__ == "__main__":
    main()
