"""Top-level online active-learning workflow."""

from __future__ import annotations

import csv
import json
import pickle
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from ase import io
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.units import fs

from .calculator import VisNetEIPASECalculator
from .config import ActiveLearningConfig
from .cp2k_labeling import label_frames
from .dataset import append_labeled_frames
from .retrain import run_retraining
from .selection import is_candidate_novel, select_representatives
from .types import CandidateFrame, WorkflowState
from .udd import calibrate_reference, threshold_value


STATE_FILE = "workflow_state.json"


def run_active_learning(config: ActiveLearningConfig) -> dict[str, Any]:
    config.validate()
    run_dir = config.output.resolved_run_dir()
    state = _prepare_state(config, run_dir)
    while state.next_round <= config.md.max_rounds:
        round_dir = run_dir / f"round_{state.next_round:04d}"
        round_dir.mkdir(parents=True, exist_ok=False)
        atoms = _load_atoms(config, state)
        round_result = _run_round(
            config=config,
            atoms=atoms,
            checkpoint_path=Path(state.current_checkpoint),
            round_index=state.next_round,
            start_global_step=state.total_steps,
            round_dir=round_dir,
        )
        labeled_frames = []
        dataset_path = Path(state.current_dataset)
        checkpoint_path = Path(state.current_checkpoint)
        restart_path = round_dir / "restart_state" / "restart_state.traj"
        if round_result["selected_frames"]:
            cp2k_dir = round_dir / "cp2k_jobs"
            labeled_frames = label_frames(round_result["selected_frames"], config=config.cp2k, output_dir=cp2k_dir)
            shutil.copy2(cp2k_dir / "labeled_frames.pkl", round_dir / "labeled_frames.pkl")
            shutil.copy2(cp2k_dir / "labeled_frames.json", round_dir / "labeled_frames.json")
            augmented_dataset = append_labeled_frames(
                dataset_path,
                labeled_frames,
                round_dir / "augmented_training.pkl",
                traj_name=f"active_round_{state.next_round:04d}",
            )
            retrain_dir = round_dir / "retrain"
            checkpoint_path = run_retraining(
                config.retrain,
                dataset_path=augmented_dataset,
                init_checkpoint=checkpoint_path,
                output_dir=retrain_dir,
            )
            dataset_path = augmented_dataset
        state.history.append(
            {
                "round": state.next_round,
                "round_dir": str(round_dir),
                "candidate_count": len(round_result["candidate_pool"]),
                "selected_count": len(round_result["selected_frames"]),
                "labeled_count": len(labeled_frames),
                "u_ref_ev_per_angstrom": round_result["u_ref_ev_per_angstrom"],
                "u_scale_ev_per_angstrom": round_result["u_scale_ev_per_angstrom"],
                "restart_state": str(restart_path),
                "checkpoint": str(checkpoint_path),
                "dataset": str(dataset_path),
            }
        )
        state.current_checkpoint = str(checkpoint_path)
        state.current_dataset = str(dataset_path)
        state.restart_state = str(restart_path)
        state.total_steps = round_result["end_global_step"]
        state.next_round += 1
        if not config.output.keep_intermediates and (round_dir / "cp2k_jobs").exists():
            for child in (round_dir / "cp2k_jobs").iterdir():
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
        _write_state(run_dir, state)
        if not round_result["selected_frames"]:
            break
    summary = state.to_dict()
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def _prepare_state(config: ActiveLearningConfig, run_dir: Path) -> WorkflowState:
    state_path = run_dir / STATE_FILE
    if config.output.overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
    if state_path.exists():
        if not config.output.resume:
            raise FileExistsError(f"Run directory already exists: {run_dir}")
        data = json.loads(state_path.read_text(encoding="utf-8"))
        return WorkflowState(**data)
    run_dir.mkdir(parents=True, exist_ok=True)
    state = WorkflowState(
        next_round=1,
        current_checkpoint=str(config.model.checkpoint),
        current_dataset=str(config.retrain.base_data),
        restart_state=None,
        total_steps=0,
    )
    _write_state(run_dir, state)
    return state


def _write_state(run_dir: Path, state: WorkflowState) -> None:
    with (run_dir / STATE_FILE).open("w", encoding="utf-8") as handle:
        json.dump(state.to_dict(), handle, indent=2, ensure_ascii=False)


def _load_atoms(config: ActiveLearningConfig, state: WorkflowState):
    if state.restart_state is not None and Path(state.restart_state).exists():
        atoms = io.read(state.restart_state, index=-1)
        atoms.set_pbc((True, True, True))
        return atoms
    atoms = io.read(config.md.initial_xyz, index=config.md.initial_frame)
    cell = np.asarray(config.md.cell, dtype=np.float64)
    if cell.shape == (3,):
        cell = np.diag(cell)
    if cell.shape != (3, 3):
        raise ValueError(f"Unsupported md.cell shape: {cell.shape}")
    atoms.set_cell(cell)
    atoms.set_pbc((True, True, True))
    if config.md.seed is not None:
        rng = np.random.default_rng(config.md.seed)
        np.random.seed(config.md.seed)
        random.seed(config.md.seed)
    else:
        rng = np.random.default_rng()
    MaxwellBoltzmannDistribution(atoms, temperature_K=config.md.temperature_k, rng=rng)
    Stationary(atoms)
    ZeroRotation(atoms)
    return atoms


def _run_round(
    *,
    config: ActiveLearningConfig,
    atoms,
    checkpoint_path: Path,
    round_index: int,
    start_global_step: int,
    round_dir: Path,
) -> dict[str, Any]:
    calc = VisNetEIPASECalculator(
        str(checkpoint_path),
        device=config.model.device,
        mode="physical",
        lambda_bias_ev=config.udd.lambda_bias_ev,
        aggregator_power=config.udd.aggregator_power,
        eps=config.udd.epsilon,
    )
    atoms.calc = calc
    dyn = Langevin(
        atoms,
        config.md.timestep_fs * fs,
        temperature_K=config.md.temperature_k,
        friction=config.md.friction,
    )
    traj_writer = Trajectory(str(round_dir / "trajectory.traj"), "w", atoms)
    logs: list[dict[str, Any]] = []
    warmup_values: list[float] = []
    candidate_pool: list[CandidateFrame] = []
    consecutive_hits = 0
    last_candidate_step = -10**9
    current_step = start_global_step
    trigger_threshold = None

    for local_step in range(1, config.md.steps_per_round + 1):
        dyn.run(1)
        current_step += 1
        results = dict(atoms.calc.results)
        if local_step % config.md.trajectory_interval == 0:
            traj_writer.write(atoms)
        current_mode = "physical" if local_step <= config.md.warmup_steps else "udd"
        if local_step % config.md.log_interval == 0:
            logs.append(_log_row(local_step, current_step, atoms, results, mode=current_mode, selected=False))
        if local_step % config.md.eval_interval == 0:
            if local_step <= config.md.warmup_steps:
                warmup_values.append(float(results["uncertainty_frame_ev_per_angstrom"]))
            else:
                if trigger_threshold is None:
                    raise RuntimeError("trigger_threshold was not set after warmup")
                u_frame = float(results["uncertainty_frame_ev_per_angstrom"])
                if u_frame > trigger_threshold:
                    consecutive_hits += 1
                else:
                    consecutive_hits = 0
                if consecutive_hits >= config.udd.threshold_hits and current_step - last_candidate_step >= config.selection.cooldown_steps:
                    candidate = _build_candidate_frame(
                        atoms,
                        results,
                        round_index=round_index,
                        source_step=current_step,
                    )
                    if is_candidate_novel(candidate.frame_feature, candidate_pool, config.selection.min_feature_distance):
                        candidate_pool.append(candidate)
                        last_candidate_step = current_step
                    consecutive_hits = 0
                    if len(candidate_pool) >= config.selection.batch_query_size:
                        logs.append(_log_row(local_step, current_step, atoms, results, mode="udd", selected=True))
                        break
        if local_step == config.md.warmup_steps:
            u_ref, u_scale = calibrate_reference(
                warmup_values,
                quantile=config.udd.reference_quantile,
                eps=config.udd.epsilon,
            )
            trigger_threshold = threshold_value(u_ref, u_scale, config.udd.threshold_sigma)
            calc.set_udd_calibration(u_ref, u_scale)
            calc.set_mode("udd")

    traj_writer.write(atoms)
    traj_writer.close()
    restart_dir = round_dir / "restart_state"
    restart_dir.mkdir(parents=True, exist_ok=True)
    io.write(restart_dir / "restart_state.traj", atoms)

    u_ref, u_scale = calibrate_reference(
        warmup_values,
        quantile=config.udd.reference_quantile,
        eps=config.udd.epsilon,
    )
    if candidate_pool:
        summary = select_representatives(candidate_pool, config.selection.batch_query_size)
        selected_frames = [candidate_pool[idx] for idx in summary.selected_indices]
    else:
        summary = None
        selected_frames = []

    _write_logs(round_dir / "md_log.csv", logs)
    _write_candidates(round_dir / "candidate_pool.pkl", candidate_pool)
    _write_candidates(round_dir / "selected_frames.pkl", selected_frames)
    with (round_dir / "round_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "round": round_index,
                "checkpoint": str(checkpoint_path),
                "warmup_samples": len(warmup_values),
                "u_ref_ev_per_angstrom": u_ref,
                "u_scale_ev_per_angstrom": u_scale,
                "trigger_threshold_ev_per_angstrom": threshold_value(u_ref, u_scale, config.udd.threshold_sigma),
                "candidate_count": len(candidate_pool),
                "selected_count": len(selected_frames),
                "selected_ids": [item.candidate_id for item in selected_frames],
                "selection_summary": None if summary is None else {
                    "selected_indices": summary.selected_indices,
                    "selected_ids": summary.selected_ids,
                    "distances": summary.distances,
                },
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    return {
        "candidate_pool": candidate_pool,
        "selected_frames": selected_frames,
        "u_ref_ev_per_angstrom": u_ref,
        "u_scale_ev_per_angstrom": u_scale,
        "end_global_step": current_step,
    }


def _build_candidate_frame(atoms, results: dict[str, Any], *, round_index: int, source_step: int) -> CandidateFrame:
    return CandidateFrame(
        candidate_id=f"round{round_index:04d}_step{source_step:08d}",
        source_round=round_index,
        source_step=source_step,
        species=atoms.get_chemical_symbols(),
        positions=np.asarray(atoms.get_positions(wrap=False), dtype=np.float64),
        cell=np.asarray(atoms.cell.array, dtype=np.float64),
        u_frame_ev_per_angstrom=float(results["uncertainty_frame_ev_per_angstrom"]),
        u_atom_ev_per_angstrom=np.asarray(results["uncertainty_per_atom_ev_per_angstrom"], dtype=np.float64),
        frame_feature=np.asarray(results["frame_feature"], dtype=np.float32),
        metadata={
            "temperature_k": float(atoms.get_temperature()),
            "mode": results.get("mode"),
            "udd_bias_energy_ev": float(results.get("udd_bias_energy_ev", 0.0)),
            "physical_energy_hartree": float(results.get("physical_energy_hartree", 0.0)),
            "physical_energy_ev": float(results.get("physical_energy_ev", 0.0)),
            "physical_forces_hartree_per_angstrom": np.asarray(
                results.get("physical_forces_hartree_per_angstrom"),
                dtype=np.float64,
            ),
        },
    )


def _write_candidates(path: Path, candidates: list[CandidateFrame]) -> None:
    with path.open("wb") as handle:
        pickle.dump([item.to_dict() for item in candidates], handle, protocol=pickle.HIGHEST_PROTOCOL)
    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "candidate_id",
                "source_round",
                "source_step",
                "u_frame_ev_per_angstrom",
                "temperature_k",
                "mode",
            ],
        )
        writer.writeheader()
        for item in candidates:
            writer.writerow(
                {
                    "candidate_id": item.candidate_id,
                    "source_round": item.source_round,
                    "source_step": item.source_step,
                    "u_frame_ev_per_angstrom": item.u_frame_ev_per_angstrom,
                    "temperature_k": item.metadata.get("temperature_k"),
                    "mode": item.metadata.get("mode"),
                }
            )


def _write_logs(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "local_step",
                "global_step",
                "mode",
                "energy_ev",
                "temperature_k",
                "uncertainty_frame_ev_per_angstrom",
                "udd_bias_energy_ev",
                "selected",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _log_row(local_step: int, global_step: int, atoms, results: dict[str, Any], *, mode: str, selected: bool) -> dict[str, Any]:
    return {
        "local_step": local_step,
        "global_step": global_step,
        "mode": mode,
        "energy_ev": results.get("energy"),
        "temperature_k": float(atoms.get_temperature()),
        "uncertainty_frame_ev_per_angstrom": results.get("uncertainty_frame_ev_per_angstrom"),
        "udd_bias_energy_ev": results.get("udd_bias_energy_ev", 0.0),
        "selected": selected,
    }
