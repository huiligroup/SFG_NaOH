"""End-to-end LQVE workflow runner."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np

from lqve.dvr import build_reference_dvr
from lqve.dvr.potential import PotentialSurface
from lqve.dvr.result import DVRResult
from lqve.embedding import EmbeddingConfig, build_embedded_geometries
from lqve.qc import QCConfig, run_qc
from lqve.perturbation import ShiftConfig, run_shift
from lqve.reference_selection import ReferenceSelectionConfig, select_reference

from .config import ReferenceSpec, TrajectorySpec, WorkflowConfig


@dataclass
class WorkflowResult:
    run_name: str
    output_dir: Path
    reference_library: Path
    frame_registry: Path
    shift_dir: Path
    all_frequencies_csv: Path
    frame_count: int


def run_lqve_workflow(config: WorkflowConfig) -> WorkflowResult:
    config.validate()
    start = time.perf_counter()
    lqve_root = config.resolved_lqve_root()
    run_root = lqve_root / "data" / "workflows" / config.run_name
    if config.overwrite:
        _remove_owned_run_dirs(lqve_root, config.run_name)
    run_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_root / "workflow_config.json", config.to_dict())

    reference_entries = [_prepare_reference(config, reference) for reference in config.references]
    reference_library = run_root / "reference_library.json"
    _write_json(
        reference_library,
        {
            "feature_type": "visnet_scalar_x_before_energy_head",
            "checkpoint": str(config.reference_selection.checkpoint or config.energy.checkpoint or ""),
            "references": reference_entries,
        },
    )

    frame_records: list[dict[str, Any]] = []
    processed = 0
    for trajectory in config.trajectories:
        for frame_index in _iter_frame_indices(trajectory):
            if config.limit_frames is not None and processed >= config.limit_frames:
                break
            record = _run_frame(
                config=config,
                trajectory=trajectory,
                frame_index=frame_index,
                reference_entries=reference_entries,
                reference_library=reference_library,
            )
            frame_records.append(record)
            processed += 1
        if config.limit_frames is not None and processed >= config.limit_frames:
            break

    frame_registry = run_root / "frame_registry.json"
    _write_json(frame_registry, {"frames": frame_records})
    shift_dir = lqve_root / "data" / "shifts" / config.run_name
    shift_results = run_shift(
        ShiftConfig(
            qc_outputs=lqve_root / "data" / "qc_outputs" / config.run_name,
            output_dir=shift_dir,
            reference_key=config.shift.reference_key,
            reference_mode="file",
            n_contract=config.shift.n_contract,
            n_transitions=config.shift.n_transitions,
            subtract_mean=config.shift.subtract_mean,
            limit_frames=config.limit_frames,
        )
    )
    all_csv = shift_dir / "all_frequencies.csv"
    _write_all_frequencies(shift_dir / "shifts_summary.csv", frame_records, all_csv)

    if not config.keep_intermediates:
        _cleanup_intermediates(lqve_root, config.run_name)

    _write_json(
        run_root / "workflow_result.json",
        {
            "run_name": config.run_name,
            "frame_count": len(frame_records),
            "shift_frame_count": len(shift_results),
            "reference_library": str(reference_library),
            "frame_registry": str(frame_registry),
            "shift_dir": str(shift_dir),
            "all_frequencies_csv": str(all_csv),
            "elapsed_seconds": time.perf_counter() - start,
        },
    )
    return WorkflowResult(
        run_name=config.run_name,
        output_dir=run_root,
        reference_library=reference_library,
        frame_registry=frame_registry,
        shift_dir=shift_dir,
        all_frequencies_csv=all_csv,
        frame_count=len(frame_records),
    )


def _prepare_reference(config: WorkflowConfig, reference: ReferenceSpec) -> dict[str, Any]:
    lqve_root = config.resolved_lqve_root()
    dvr_dir = Path(reference.dvr_data) if reference.dvr_data is not None else lqve_root / "data" / "dvr" / reference.reference_id
    energies_path = (
        Path(reference.reference_energies)
        if reference.reference_energies is not None
        else lqve_root / "references" / "pes" / f"{reference.reference_id}_reference_energies_hartree.npy"
    )
    dvr_npz = dvr_dir / "dvr_result.npz"
    if config.overwrite and dvr_dir.exists():
        shutil.rmtree(dvr_dir)
    if not dvr_npz.exists():
        result = build_reference_dvr(reference.pes, reference.dvr)
        result.save(dvr_dir)
    else:
        result = DVRResult.load(dvr_dir)
    if config.overwrite or not energies_path.exists():
        energies = _evaluate_reference_energies(reference, result)
        energies_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(energies_path, energies)
    entry = {
        "reference_id": reference.reference_id,
        "reference_xyz": str(reference.reference_xyz),
        "modes": str(reference.modes),
        "dvr_data": str(dvr_dir),
        "reference_energies": str(energies_path),
        "grid_unit": reference.grid_unit,
        "mode_unit": reference.mode_unit,
        "pes": str(reference.pes),
        **reference.metadata,
    }
    if reference.feature_file is not None:
        entry["feature_file"] = str(reference.feature_file)
    return entry


def _evaluate_reference_energies(reference: ReferenceSpec, dvr_result: DVRResult) -> np.ndarray:
    potential = PotentialSurface.from_file(
        reference.pes,
        dims=reference.dvr.dims,
        coord_unit=reference.dvr.coord_unit,
        energy_unit=reference.dvr.energy_unit,
    )
    mesh = np.meshgrid(*[np.asarray(grid, dtype=np.float64) for grid in dvr_result.grids_bohr], indexing="ij")
    points = np.stack([axis.reshape(-1) for axis in mesh], axis=1)
    return potential.evaluate(points).astype(np.float64, copy=False)


def _run_frame(
    config: WorkflowConfig,
    trajectory: TrajectorySpec,
    frame_index: int,
    reference_entries: list[dict[str, Any]],
    reference_library: Path,
) -> dict[str, Any]:
    lqve_root = config.resolved_lqve_root()
    frame_id = f"{trajectory.traj_id}_frame_{frame_index:06d}"
    embedding_dir = lqve_root / "data" / "embedded_geometries" / config.run_name / frame_id
    qc_root = lqve_root / "data" / "qc_outputs" / config.run_name
    selected = _select_reference_for_frame(config, trajectory, frame_index, reference_entries, reference_library, embedding_dir)
    probe_indices = trajectory.probe_indices or selected["query_metadata"]["probe_indices"]
    embedding_metadata = {
        "frame_id": frame_id,
        "traj_id": trajectory.traj_id,
        "frame_index": frame_index,
        "selected_reference": selected,
        "reference_energies": selected["selected"]["reference_energies"],
    }
    if trajectory.cell is not None:
        embedding_metadata["cell"] = trajectory.cell
    elif config.energy.cell is not None:
        embedding_metadata["cell"] = config.energy.cell

    if config.overwrite and embedding_dir.exists():
        shutil.rmtree(embedding_dir)
    if not (config.resume and (embedding_dir / "embedded_geometries.npz").exists()):
        result = build_embedded_geometries(
            EmbeddingConfig(
                reference_xyz=selected["selected"]["reference_xyz"],
                system_xyz=trajectory.path,
                probe_indices=probe_indices,
                modes=selected["selected"]["modes"],
                dvr_data=selected["selected"]["dvr_data"],
                output_dir=embedding_dir,
                grid_unit=selected["selected"].get("grid_unit", config.embedding.grid_unit),
                mode_unit=selected["selected"].get("mode_unit", config.embedding.mode_unit),
                frame_index=frame_index,
                centroid_weight=config.embedding.centroid_weight,
                write_xyz=config.embedding.write_xyz,
                prefix=config.embedding.prefix,
                metadata=embedding_metadata,
            )
        )
        result.save(embedding_dir, write_xyz_files=config.embedding.write_xyz, prefix=config.embedding.prefix)

    run_qc(
        QCConfig(
            geometries=embedding_dir,
            output_dir=qc_root,
            backend=config.energy.backend,
            checkpoint=config.energy.checkpoint,
            device=config.energy.device,
            cell=trajectory.cell or config.energy.cell,
            batch_size=config.energy.batch_size,
            workers=config.energy.workers,
            limit_geometries=config.energy.limit_geometries,
            command=config.energy.command,
            template=config.energy.template,
            nproc=config.energy.nproc,
            method=config.energy.method,
            keep_workdirs=config.energy.keep_workdirs,
            work_dir=config.energy.work_dir,
            resume=config.resume,
        )
    )
    return {
        "frame_id": frame_id,
        "traj_id": trajectory.traj_id,
        "frame_index": frame_index,
        "trajectory": str(trajectory.path),
        "embedding_dir": str(embedding_dir),
        "qc_root": str(qc_root),
        "reference_id": selected["selected"].get("reference_id", ""),
    }


def _select_reference_for_frame(
    config: WorkflowConfig,
    trajectory: TrajectorySpec,
    frame_index: int,
    reference_entries: list[dict[str, Any]],
    reference_library: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if config.dynamic_reference:
        selection = select_reference(
            ReferenceSelectionConfig(
                reference_library=reference_library,
                checkpoint=config.reference_selection.checkpoint,
                system_xyz=trajectory.path,
                feature_data=trajectory.feature_data,
                frame_index=frame_index,
                mol_id=trajectory.mol_id,
                probe_indices=trajectory.probe_indices,
                cell=trajectory.cell or config.energy.cell,
                device=config.reference_selection.device,
                top_k=config.reference_selection.top_k,
                output_dir=output_dir,
            )
        )
        if not selection.lqve_ready:
            raise ValueError(f"Selected reference is not LQVE-ready: {selection.missing_assets}")
        return selection.to_dict()

    reference_id = trajectory.reference_id or reference_entries[0]["reference_id"]
    selected = next((item for item in reference_entries if item["reference_id"] == reference_id), None)
    if selected is None:
        raise ValueError(f"Unknown fixed reference_id {reference_id!r} for trajectory {trajectory.traj_id}")
    return {
        "query_metadata": {
            "probe_indices": trajectory.probe_indices,
            "mol_id": trajectory.mol_id,
            "frame_index": frame_index,
            "traj_id": trajectory.traj_id,
        },
        "selected": selected,
        "candidates": [selected],
        "lqve_ready": True,
        "missing_assets": [],
        "metadata": {"distance": "fixed_reference", "reference_library": str(reference_library)},
    }


def _iter_frame_indices(trajectory: TrajectorySpec) -> list[int]:
    stop = trajectory.frame_stop
    if stop is None:
        stop = _count_xyz_frames(trajectory.path)
    return list(range(trajectory.frame_start, stop, trajectory.frame_stride))


def _count_xyz_frames(path: str | Path) -> int:
    path = Path(path)
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        while True:
            first = handle.readline()
            while first and not first.strip():
                first = handle.readline()
            if not first:
                return count
            n_atoms = int(first.strip())
            comment = handle.readline()
            if not comment:
                raise ValueError(f"Missing comment line in XYZ frame {count}: {path}")
            for _ in range(n_atoms):
                if not handle.readline():
                    raise ValueError(f"Unexpected EOF in XYZ frame {count}: {path}")
            count += 1


def _write_all_frequencies(summary_csv: Path, frame_records: list[dict[str, Any]], output_csv: Path) -> None:
    with summary_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    registry = {record["frame_id"]: record for record in frame_records}
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_csv.write_text("", encoding="utf-8")
        return
    fieldnames = ["traj_id", "frame_index"] + list(rows[0].keys())
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record = registry.get(row.get("frame_id", ""), {})
            writer.writerow(
                {
                    "traj_id": record.get("traj_id", ""),
                    "frame_index": record.get("frame_index", ""),
                    **row,
                }
            )


def _cleanup_intermediates(lqve_root: Path, run_name: str) -> None:
    for path in (
        lqve_root / "data" / "embedded_geometries" / run_name,
        lqve_root / "data" / "qc_outputs" / run_name,
    ):
        if path.exists():
            shutil.rmtree(path)


def _remove_owned_run_dirs(lqve_root: Path, run_name: str) -> None:
    for path in (
        lqve_root / "data" / "workflows" / run_name,
        lqve_root / "data" / "embedded_geometries" / run_name,
        lqve_root / "data" / "qc_outputs" / run_name,
        lqve_root / "data" / "shifts" / run_name,
    ):
        if path.exists():
            shutil.rmtree(path)


def _write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(data), handle, indent=2, ensure_ascii=False)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
