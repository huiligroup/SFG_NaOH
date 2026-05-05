"""Frame/chunk orchestration for LQVE energy backends."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time
from typing import Any

import numpy as np

from .config import QCConfig
from .io import append_manifest, load_geometry_batches, parse_cell, save_energy_result
from .types import EnergyResult, GeometryBatch, concatenate_results


def run_qc(config: QCConfig) -> list[EnergyResult]:
    config.validate()
    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    cell = parse_cell(config.cell)
    batches = load_geometry_batches(
        config.geometries,
        cell=cell,
        limit_geometries=config.limit_geometries,
    )
    if config.limit_frames is not None:
        batches = batches[: config.limit_frames]

    if config.dry_run:
        rows = [
            {
                "frame_id": batch.frame_id,
                "backend": config.backend,
                "geometry_count": batch.geometry_count,
                "dry_run": True,
            }
            for batch in batches
        ]
        append_manifest(manifest_path, rows)
        return []

    results: list[EnergyResult] = []
    for batch in batches:
        frame_output = output_dir / _safe_frame_name(batch.frame_id)
        if config.resume and (frame_output / "qc_energies.npz").exists():
            append_manifest(
                manifest_path,
                [{"frame_id": batch.frame_id, "backend": config.backend, "status": "skipped_existing"}],
            )
            continue
        start = time.perf_counter()
        result = _evaluate_frame(config, batch)
        result.metadata.setdefault("frame_id", batch.frame_id)
        result.metadata.setdefault("input_embedding", batch.metadata.get("embedding_dir"))
        if "selected_reference" in batch.metadata:
            result.metadata.setdefault("selected_reference", batch.metadata["selected_reference"])
        if "reference_energies" in batch.metadata:
            result.metadata.setdefault("reference_energies", batch.metadata["reference_energies"])
        result.metadata["frame_elapsed_seconds"] = time.perf_counter() - start
        save_energy_result(result, frame_output)
        _write_frame_manifest(manifest_path, result, frame_output)
        results.append(result)
    return results


def _evaluate_frame(config: QCConfig, batch: GeometryBatch) -> EnergyResult:
    chunks = list(_iter_chunks(batch, config.batch_size))
    if _can_parallelize(config) and len(chunks) > 1:
        try:
            with ProcessPoolExecutor(max_workers=config.workers) as pool:
                futures = [
                    pool.submit(_evaluate_chunk_worker, config.to_dict(), chunk)
                    for chunk in chunks
                ]
                results = [future.result() for future in as_completed(futures)]
            results.sort(key=lambda item: int(item.source_indices[0]) if item.source_indices is not None else 0)
            return _merge_chunk_results(results)
        except (OSError, PermissionError) as exc:
            fallback_note = f"process_pool_unavailable: {exc}; falling back to serial execution"
        else:
            fallback_note = ""
    else:
        fallback_note = ""

    backend = build_backend(config)
    results = [backend.evaluate(chunk) for chunk in chunks]
    merged = _merge_chunk_results(results)
    if fallback_note:
        merged.metadata["parallel_fallback"] = fallback_note
    return merged


def _evaluate_chunk_worker(config_dict: dict[str, Any], batch: GeometryBatch) -> EnergyResult:
    config = QCConfig.from_dict(config_dict)
    backend = build_backend(config)
    return backend.evaluate(batch)


def build_backend(config: QCConfig) -> Any:
    work_dir = config.resolved_work_dir()
    if config.backend == "mock":
        from .backends import MockBackend

        return MockBackend()
    if config.backend == "xtb":
        from .backends import XTBBackend

        return XTBBackend(
            command=config.command or f"xtb {config.method}",
            work_dir=work_dir,
            template=config.template,
            keep_workdirs=config.keep_workdirs,
        )
    if config.backend == "gaussian":
        from .backends import GaussianBackend

        return GaussianBackend(
            command=config.command or "g16",
            work_dir=work_dir,
            template=config.template,
            keep_workdirs=config.keep_workdirs,
        )
    if config.backend == "cp2k":
        from .backends import CP2KBackend

        return CP2KBackend(
            command=config.command or "cp2k.popt",
            work_dir=work_dir,
            template=config.template,
            nproc=config.nproc,
            keep_workdirs=config.keep_workdirs,
        )
    if config.backend == "visnet":
        from .backends import VisNetBackend

        return VisNetBackend(
            checkpoint=config.checkpoint,
            device=config.device,
            batch_size=config.batch_size,
            cell=parse_cell(config.cell),
        )
    raise ValueError(f"Unsupported backend: {config.backend}")


def _iter_chunks(batch: GeometryBatch, chunk_size: int) -> list[GeometryBatch]:
    return [
        batch.slice(start, min(start + chunk_size, batch.geometry_count))
        for start in range(0, batch.geometry_count, chunk_size)
    ]


def _merge_chunk_results(results: list[EnergyResult]) -> EnergyResult:
    merged = concatenate_results(results)
    manifests = []
    for result in results:
        manifests.extend(result.metadata.get("manifest", []))
    if manifests:
        merged.metadata["manifest"] = manifests
    merged.metadata["chunk_count"] = len(results)
    return merged


def _can_parallelize(config: QCConfig) -> bool:
    return config.workers > 1 and config.backend in {"xtb", "gaussian", "cp2k", "mock"}


def _write_frame_manifest(path: Path, result: EnergyResult, frame_output: Path) -> None:
    rows = result.metadata.get("manifest")
    if rows:
        append_manifest(path, rows)
    append_manifest(
        path,
        [
            {
                "frame_id": result.frame_id,
                "backend": result.backend,
                "status": "done",
                "geometry_count": int(result.energies_hartree.size),
                "success_count": int(result.success_mask.sum()),
                "output_dir": str(frame_output),
            }
        ],
    )


def _safe_frame_name(frame_id: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(frame_id))
    return text or "frame"
