"""Runner for LQVE shift calculations."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from lqve.dvr.result import DVRResult

from .config import ShiftConfig
from .io import (
    append_manifest,
    load_qc_output,
    load_reference_energies,
    resolve_qc_output_dirs,
    write_summary_csv,
)
from .result import ShiftResult
from .shift import compute_shift


def run_shift(config: ShiftConfig) -> list[ShiftResult]:
    config.validate()
    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    qc_dirs = resolve_qc_output_dirs(config.qc_outputs)
    if config.limit_frames is not None:
        qc_dirs = qc_dirs[: config.limit_frames]
    dvr_cache: dict[str, DVRResult] = {}
    reference_cache: dict[tuple[str, str | None], np.ndarray] = {}
    first_qc_reference = (
        np.asarray(load_qc_output(qc_dirs[0])["energies_hartree"], dtype=np.float64)
        if config.reference_mode == "first-qc"
        else None
    )

    results: list[ShiftResult] = []
    rows: list[dict[str, Any]] = []
    reference_ids: list[str] = []
    transitions_parts: list[np.ndarray] = []
    shifts_parts: list[np.ndarray] = []
    start_all = time.perf_counter()
    for qc_dir in qc_dirs:
        start = time.perf_counter()
        qc_output = load_qc_output(qc_dir)
        frame_reference = _frame_reference_metadata(qc_output)
        reference_id = str(frame_reference.get("reference_id", "")) if frame_reference else ""
        dvr_data: Path | None = None
        reference_path: Path | None = None
        try:
            dvr_data = _resolve_frame_dvr_data(config, frame_reference)
            reference_path = _resolve_frame_reference_energies(config, frame_reference)
            dvr_result = _load_dvr_cached(dvr_data, dvr_cache)
            if config.reference_mode == "file":
                reference = _load_reference_cached(reference_path, config.reference_key, reference_cache)
            else:
                reference = first_qc_reference
                if reference is None:
                    raise ValueError("reference-mode first-qc could not load first QC frame")
            result = compute_shift(dvr_result, qc_output, reference, config)
            status = "done" if result.success else "failed_grid"
            error = ""
        except Exception as exc:  # noqa: BLE001 - keep frame-level failure in manifest.
            result = _failed_result(qc_output, config, exc)
            status = "error"
            error = str(exc)
        frame_output = output_dir / _safe_frame_name(result.frame_id)
        result.metadata["frame_elapsed_seconds"] = time.perf_counter() - start
        result.metadata["dvr_data"] = "" if dvr_data is None else str(dvr_data)
        result.metadata["reference_energies"] = "" if reference_path is None else str(reference_path)
        result.metadata["reference_id"] = reference_id
        if frame_reference:
            result.metadata["selected_reference"] = frame_reference
        result.save(frame_output)
        results.append(result)
        rows.append(_summary_row(result, config.n_transitions))
        reference_ids.append(reference_id)
        transitions_parts.append(result.transitions_cm1.reshape(1, -1))
        shifts_parts.append(result.shifts_cm1.reshape(1, -1))
        append_manifest(
            manifest_path,
            [
                {
                    "frame_id": result.frame_id,
                    "status": status,
                    "success": result.success,
                    "source_qc_dir": result.source_qc_dir,
                    "output_dir": str(frame_output),
                    "reference_id": reference_id,
                    "dvr_data": "" if dvr_data is None else str(dvr_data),
                    "error": error,
                }
            ],
        )

    write_summary_csv(output_dir / "shifts_summary.csv", rows, config.n_transitions)
    np.savez_compressed(
        output_dir / "shifts_summary.npz",
        frame_id=np.asarray([result.frame_id for result in results]),
        transitions_cm1=np.concatenate(transitions_parts, axis=0) if transitions_parts else np.empty((0, 0)),
        shifts_cm1=np.concatenate(shifts_parts, axis=0) if shifts_parts else np.empty((0, 0)),
        success=np.asarray([result.success for result in results], dtype=bool),
        reference_id=np.asarray(reference_ids, dtype=object),
    )
    metadata = {
        "config": config.to_dict(),
        "reference_mode_warning": (
            "reference-mode first-qc is intended for smoke/debug runs only"
            if config.reference_mode == "first-qc"
            else ""
        ),
        "frame_count": len(results),
        "success_count": int(sum(result.success for result in results)),
        "reference_counts": _counts(reference_ids),
        "elapsed_seconds": time.perf_counter() - start_all,
        "files": {
            "summary_csv": "shifts_summary.csv",
            "summary_arrays": "shifts_summary.npz",
            "manifest": "manifest.jsonl",
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    return results


def _failed_result(qc_output: dict[str, Any], config: ShiftConfig, exc: Exception) -> ShiftResult:
    return ShiftResult(
        frame_id=str(qc_output.get("frame_id", "unknown")),
        levels_hartree=np.full(config.n_contract, np.nan, dtype=np.float64),
        transitions_cm1=np.full(config.n_transitions, np.nan, dtype=np.float64),
        shifts_cm1=np.full(config.n_transitions, np.nan, dtype=np.float64),
        success_mask=np.asarray(qc_output.get("success_mask", []), dtype=bool),
        grid_points=np.asarray(qc_output.get("grid_points", []), dtype=np.float64),
        source_qc_dir=str(qc_output.get("qc_dir", "")),
        metadata={"error": str(exc), "method": "effective_hamiltonian"},
    )


def _summary_row(result: ShiftResult, n_transitions: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "frame_id": result.frame_id,
        "success": result.success,
        "source_qc_dir": result.source_qc_dir,
        "reference_id": result.metadata.get("reference_id", ""),
        "dvr_data": result.metadata.get("dvr_data", ""),
        "reference_energies": result.metadata.get("reference_energies", ""),
    }
    for idx in range(n_transitions):
        row[f"transition_{idx + 1}_cm1"] = float(result.transitions_cm1[idx])
        row[f"shift_{idx + 1}_cm1"] = float(result.shifts_cm1[idx])
    return row


def _safe_frame_name(frame_id: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(frame_id))
    return text or "frame"


def _frame_reference_metadata(qc_output: dict[str, Any]) -> dict[str, Any]:
    metadata = qc_output.get("metadata", {})
    selected = metadata.get("selected_reference")
    if isinstance(selected, dict):
        if "selected" in selected and isinstance(selected["selected"], dict):
            return dict(selected["selected"])
        return dict(selected)
    input_embedding = metadata.get("input_embedding")
    if input_embedding:
        path = Path(input_embedding) / "selected_reference.json"
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if "selected" in loaded and isinstance(loaded["selected"], dict):
                return dict(loaded["selected"])
            return dict(loaded)
    return {}


def _resolve_frame_dvr_data(config: ShiftConfig, frame_reference: dict[str, Any]) -> Path:
    if config.dvr_data is not None:
        return Path(config.dvr_data)
    if frame_reference.get("dvr_data"):
        return Path(frame_reference["dvr_data"])
    raise ValueError("No dvr_data provided and selected_reference metadata does not contain dvr_data")


def _resolve_frame_reference_energies(config: ShiftConfig, frame_reference: dict[str, Any]) -> Path | None:
    if config.reference_mode != "file":
        return None
    if config.reference_energies is not None:
        return Path(config.reference_energies)
    if frame_reference.get("reference_energies"):
        return Path(frame_reference["reference_energies"])
    raise ValueError(
        "No reference_energies provided and selected_reference metadata does not contain reference_energies"
    )


def _load_dvr_cached(path: Path, cache: dict[str, DVRResult]) -> DVRResult:
    key = str(path)
    if key not in cache:
        cache[key] = DVRResult.load(path)
    return cache[key]


def _load_reference_cached(
    path: Path | None,
    key: str | None,
    cache: dict[tuple[str, str | None], np.ndarray],
) -> np.ndarray:
    if path is None:
        raise ValueError("reference energy path is required")
    cache_key = (str(path), key)
    if cache_key not in cache:
        cache[cache_key] = load_reference_energies(path, key=key)
    return cache[cache_key]


def _counts(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = value or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return counts
