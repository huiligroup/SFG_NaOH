#!/usr/bin/env python
"""Embed reference DVR points into real-time molecular environments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = THIS_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lqve.embedding import EmbeddingConfig, build_embedded_geometries
from lqve.reference_selection import ReferenceSelectionConfig, select_reference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DVR-point geometries embedded in MD environments."
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional workflow config.")
    parser.add_argument("--reference-xyz", type=Path, default=None, help="Reference probe XYZ.")
    parser.add_argument("--system-xyz", type=Path, default=None, help="System or cluster XYZ.")
    parser.add_argument(
        "--probe-indices",
        default=None,
        help="Zero-based comma-separated probe atom indices in system XYZ.",
    )
    parser.add_argument("--modes", type=Path, default=None, help="Mode matrix file.")
    parser.add_argument(
        "--dvr-data",
        type=Path,
        default=Path("NaOH_7.5M/src/lqve/data/dvr"),
        help="Reference DVR data directory.",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=Path("NaOH_7.5M/src/lqve/data/processed"),
        help="Alias for --system-xyz kept for workflow readability.",
    )
    parser.add_argument(
        "--grid-unit",
        default="bohr",
        choices=("bohr", "angstrom"),
        help="Unit of DVR grid values loaded from --dvr-data.",
    )
    parser.add_argument(
        "--mode-unit",
        default="dimensionless",
        choices=("dimensionless", "angstrom_per_angstrom", "angstrom_per_bohr", "bohr_per_bohr"),
        help="Unit convention for mode vectors.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Frame index for multi-frame XYZ system files.",
    )
    parser.add_argument(
        "--masses",
        default=None,
        help="Optional comma-separated probe masses. Defaults to masses inferred from elements.",
    )
    parser.add_argument(
        "--centroid-weight",
        default="mass",
        choices=("mass", "sqrt_mass", "uniform"),
        help="Weights used for centroids and Kabsch alignment.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for embedded DVR geometries.",
    )
    parser.add_argument("--write-xyz", action="store_true", help="Write one XYZ per DVR grid point.")
    parser.add_argument("--prefix", default="embedded", help="Prefix for optional XYZ files.")
    parser.add_argument(
        "--reference-library",
        type=Path,
        default=None,
        help="Optional descriptor reference library. If set, selects reference_xyz/modes/dvr_data per frame.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="ViSNet checkpoint for dynamic reference selection.")
    parser.add_argument("--mol-id", type=int, default=None, help="Na12 molecule id for dynamic reference selection.")
    parser.add_argument(
        "--feature-data",
        type=Path,
        default=None,
        help="Optional NaOH_7.5M/data/visnet/*.pkl source for descriptor extraction.",
    )
    parser.add_argument(
        "--cell",
        default=None,
        help="Cell lengths/matrix for descriptor extraction from --system-xyz, e.g. 16.63,16.63,44.10.",
    )
    parser.add_argument("--device", default=None, help="Device for ViSNet descriptor extraction, e.g. cpu or cuda.")
    parser.add_argument(
        "--reference-selection-output",
        type=Path,
        default=None,
        help="Optional directory for query_feature.npy and reference_match.json.",
    )
    parser.add_argument("--reference-top-k", type=int, default=5, help="Number of nearest reference candidates to record.")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> EmbeddingConfig:
    if args.config is not None:
        with args.config.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        overrides = {
            "reference_xyz": args.reference_xyz,
            "system_xyz": args.system_xyz if args.system_xyz is not None else None,
            "modes": args.modes,
            "dvr_data": args.dvr_data if args.dvr_data != Path("NaOH_7.5M/src/lqve/data/dvr") else None,
            "output_dir": args.output_dir,
        }
        for key, value in overrides.items():
            if value is not None:
                raw[key] = str(value)
        if args.probe_indices is not None:
            raw["probe_indices"] = parse_probe_indices(args.probe_indices)
        if args.write_xyz:
            raw["write_xyz"] = True
        _apply_reference_selection_overrides(raw, args)
        raw = apply_dynamic_reference_selection(raw)
        return EmbeddingConfig.from_dict(raw)

    system_xyz = args.system_xyz
    if system_xyz is None and args.trajectory != Path("NaOH_7.5M/src/lqve/data/processed"):
        system_xyz = args.trajectory
    raw: dict[str, Any] = {
        "reference_xyz": None if args.reference_xyz is None else str(args.reference_xyz),
        "system_xyz": None if system_xyz is None else str(system_xyz),
        "probe_indices": None if args.probe_indices is None else parse_probe_indices(args.probe_indices),
        "modes": None if args.modes is None else str(args.modes),
        "dvr_data": str(args.dvr_data),
        "output_dir": None if args.output_dir is None else str(args.output_dir),
        "grid_unit": args.grid_unit,
        "mode_unit": args.mode_unit,
        "frame_index": args.frame_index,
        "masses": parse_masses(args.masses),
        "centroid_weight": args.centroid_weight,
        "write_xyz": args.write_xyz,
        "prefix": args.prefix,
    }
    _apply_reference_selection_overrides(raw, args)
    raw = apply_dynamic_reference_selection(raw)
    required = {
        "--reference-xyz": raw.get("reference_xyz"),
        "--system-xyz": system_xyz,
        "--probe-indices": raw.get("probe_indices"),
        "--modes": raw.get("modes"),
        "--dvr-data": raw.get("dvr_data"),
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")
    return EmbeddingConfig.from_dict({key: value for key, value in raw.items() if value is not None})


def _apply_reference_selection_overrides(raw: dict[str, Any], args: argparse.Namespace) -> None:
    overrides = {
        "reference_library": args.reference_library,
        "checkpoint": args.checkpoint,
        "mol_id": args.mol_id,
        "feature_data": args.feature_data,
        "cell": args.cell,
        "device": args.device,
        "reference_selection_output": args.reference_selection_output,
        "reference_top_k": args.reference_top_k,
    }
    for key, value in overrides.items():
        if value is not None:
            raw[key] = str(value) if isinstance(value, Path) else value


def apply_dynamic_reference_selection(raw: dict[str, Any]) -> dict[str, Any]:
    if not raw.get("reference_library"):
        return _strip_reference_selection_keys(raw)
    if not raw.get("checkpoint"):
        raise ValueError("--checkpoint is required when --reference-library is used")

    output_dir = raw.get("reference_selection_output") or raw.get("output_dir")
    selection_config = ReferenceSelectionConfig(
        reference_library=raw["reference_library"],
        checkpoint=raw["checkpoint"],
        system_xyz=raw.get("system_xyz"),
        feature_data=raw.get("feature_data"),
        frame_index=int(raw.get("frame_index", 0)),
        mol_id=raw.get("mol_id"),
        probe_indices=raw.get("probe_indices"),
        cell=raw.get("cell"),
        device=raw.get("device", "cpu"),
        top_k=int(raw.get("reference_top_k", 5)),
        output_dir=output_dir,
    )
    selection = select_reference(selection_config)
    if not selection.lqve_ready:
        raise ValueError(
            "Selected reference is missing LQVE assets "
            f"{selection.missing_assets}. Provide a reference_library JSON with "
            "reference_xyz, modes, dvr_data, and reference_energies for each reference."
        )
    selected = selection.selected
    raw["reference_xyz"] = selected["reference_xyz"]
    raw["modes"] = selected["modes"]
    raw["dvr_data"] = selected["dvr_data"]
    if not raw.get("probe_indices"):
        raw["probe_indices"] = selection.query_metadata["probe_indices"]
    metadata = dict(raw.get("metadata") or {})
    metadata["selected_reference"] = selection.to_dict()
    if selected.get("reference_energies"):
        metadata["reference_energies"] = selected["reference_energies"]
    raw["metadata"] = metadata
    return _strip_reference_selection_keys(raw)


def _strip_reference_selection_keys(raw: dict[str, Any]) -> dict[str, Any]:
    clean = dict(raw)
    for key in (
        "reference_library",
        "checkpoint",
        "mol_id",
        "feature_data",
        "cell",
        "reference_selection_output",
        "reference_top_k",
        "device",
    ):
        clean.pop(key, None)
    return clean


def parse_probe_indices(value: str) -> list[int]:
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not indices:
        raise ValueError("--probe-indices cannot be empty")
    return indices


def parse_masses(value: str | None) -> list[float] | None:
    if value is None:
        return None
    masses = [float(item.strip()) for item in value.split(",") if item.strip()]
    return masses or None


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    output_dir = config.resolved_output_dir()
    print(
        "building_embedded_geometries "
        f"reference={config.reference_xyz} system={config.system_xyz} "
        f"dvr_data={config.dvr_data} output_dir={output_dir}"
    )
    result = build_embedded_geometries(config)
    result.save(output_dir, write_xyz_files=config.write_xyz, prefix=config.prefix)
    print(
        "saved_embedded_geometries "
        f"arrays={output_dir / 'embedded_geometries.npz'} "
        f"metadata={output_dir / 'metadata.json'} "
        f"grid_count={len(result.grid_points)}"
    )


if __name__ == "__main__":
    main()
