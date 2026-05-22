"""CP2K-based labeling for selected active-learning frames."""

from __future__ import annotations

import json
import pickle
import re
import shlex
import subprocess
import time
from pathlib import Path

import numpy as np
from ase.units import Bohr

from .config import CP2KConfig
from .types import CandidateFrame, LabeledFrame


FORCE_AU_TO_HARTREE_PER_ANGSTROM = 1.0 / Bohr


def label_frames(
    candidates: list[CandidateFrame],
    *,
    config: CP2KConfig,
    output_dir: str | Path,
) -> list[LabeledFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.execution == "mock":
        frames = [_mock_label(frame) for frame in candidates]
    elif config.execution == "direct":
        frames = [_label_direct(frame, config, output_dir) for frame in candidates]
    else:
        frames = _label_slurm(candidates, config, output_dir)
    with (output_dir / "labeled_frames.pkl").open("wb") as handle:
        pickle.dump([frame.to_dict() for frame in frames], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with (output_dir / "labeled_frames.json").open("w", encoding="utf-8") as handle:
        json.dump([frame.to_dict() for frame in frames], handle, indent=2, ensure_ascii=False)
    return frames


def _mock_label(frame: CandidateFrame) -> LabeledFrame:
    metadata = dict(frame.metadata)
    energy = float(metadata.get("physical_energy_hartree", 0.0))
    forces = np.asarray(
        metadata.get(
            "physical_forces_hartree_per_angstrom",
            np.zeros((len(frame.species), 3), dtype=np.float64),
        ),
        dtype=np.float64,
    )
    return LabeledFrame(
        candidate_id=frame.candidate_id,
        source_round=frame.source_round,
        source_step=frame.source_step,
        species=list(frame.species),
        positions=np.asarray(frame.positions, dtype=np.float64),
        cell=np.asarray(frame.cell, dtype=np.float64),
        energy_hartree=energy,
        forces_hartree_per_angstrom=forces,
        metadata={"label_backend": "mock", **metadata},
    )


def _label_direct(frame: CandidateFrame, config: CP2KConfig, output_dir: Path) -> LabeledFrame:
    job_dir = _prepare_job_dir(frame, config, output_dir)
    cmd = _command_tokens(config.command, config.nproc, job_dir / "input.inp")
    completed = subprocess.run(cmd, cwd=job_dir, text=True, capture_output=True, check=False)
    stdout_path = job_dir / "stdout.log"
    stdout_path.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"CP2K labeling failed for {frame.candidate_id}: returncode={completed.returncode}")
    energy, forces = _parse_cp2k_outputs(job_dir, natoms=len(frame.species))
    result = LabeledFrame(
        candidate_id=frame.candidate_id,
        source_round=frame.source_round,
        source_step=frame.source_step,
        species=list(frame.species),
        positions=np.asarray(frame.positions, dtype=np.float64),
        cell=np.asarray(frame.cell, dtype=np.float64),
        energy_hartree=energy,
        forces_hartree_per_angstrom=forces,
        metadata={
            "label_backend": "cp2k_direct",
            "command": cmd,
            "job_dir": str(job_dir),
        },
    )
    if not config.keep_workdirs:
        for item in job_dir.iterdir():
            if item.name not in {"input.inp", "cp2k.out", "stdout.log", "structure.xyz"}:
                if item.is_file():
                    item.unlink()
    return result


def _label_slurm(candidates: list[CandidateFrame], config: CP2KConfig, output_dir: Path) -> list[LabeledFrame]:
    jobs: list[dict[str, object]] = []
    for frame in candidates:
        job_dir = _prepare_job_dir(frame, config, output_dir)
        script_path = job_dir / "run.slurm"
        command = " ".join(shlex.quote(token) for token in _command_tokens(config.command, config.nproc, job_dir / "input.inp"))
        lines = ["#!/bin/bash"]
        for arg in config.sbatch_args:
            lines.append(f"#SBATCH {arg}")
        lines.extend(
            [
                "set +e",
                "cd \"$SLURM_SUBMIT_DIR\"",
                f"{command}",
                "status=$?",
                "echo ${status} > exit_code.txt",
                "exit ${status}",
            ]
        )
        script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        completed = subprocess.run(
            [config.submit_command, script_path.name],
            cwd=job_dir,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"sbatch failed for {frame.candidate_id}: {completed.stderr.strip()}")
        match = re.search(r"Submitted batch job\s+(\d+)", completed.stdout)
        if match is None:
            raise RuntimeError(f"Could not parse sbatch job id for {frame.candidate_id}: {completed.stdout.strip()}")
        jobs.append({"frame": frame, "job_dir": job_dir, "job_id": match.group(1)})

    start = time.perf_counter()
    pending = set(range(len(jobs)))
    while pending:
        finished: list[int] = []
        for idx in sorted(pending):
            job_dir = Path(jobs[idx]["job_dir"])
            exit_code_path = job_dir / "exit_code.txt"
            if exit_code_path.exists():
                finished.append(idx)
        if finished:
            pending.difference_update(finished)
            continue
        if time.perf_counter() - start > config.max_wait_seconds:
            raise TimeoutError(f"Timed out waiting for SLURM CP2K jobs after {config.max_wait_seconds} seconds")
        time.sleep(config.poll_interval_seconds)

    labeled: list[LabeledFrame] = []
    for job in jobs:
        frame = job["frame"]
        job_dir = Path(job["job_dir"])
        exit_code = int((job_dir / "exit_code.txt").read_text(encoding="utf-8").strip())
        if exit_code != 0:
            raise RuntimeError(f"SLURM CP2K labeling failed for {frame.candidate_id}: exit_code={exit_code}")
        energy, forces = _parse_cp2k_outputs(job_dir, natoms=len(frame.species))
        labeled.append(
            LabeledFrame(
                candidate_id=frame.candidate_id,
                source_round=frame.source_round,
                source_step=frame.source_step,
                species=list(frame.species),
                positions=np.asarray(frame.positions, dtype=np.float64),
                cell=np.asarray(frame.cell, dtype=np.float64),
                energy_hartree=energy,
                forces_hartree_per_angstrom=forces,
                metadata={
                    "label_backend": "cp2k_slurm",
                    "job_id": str(job["job_id"]),
                    "job_dir": str(job_dir),
                },
            )
        )
    return labeled


def _prepare_job_dir(frame: CandidateFrame, config: CP2KConfig, output_dir: Path) -> Path:
    job_dir = output_dir / frame.candidate_id
    job_dir.mkdir(parents=True, exist_ok=False)
    template_text = Path(config.template_source).read_text(encoding="utf-8")
    rendered = _render_cp2k_input(template_text, frame.species, frame.positions, frame.cell, frame.candidate_id)
    (job_dir / "input.inp").write_text(rendered, encoding="utf-8")
    _write_xyz(job_dir / "structure.xyz", frame.species, frame.positions)
    with (job_dir / "candidate_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(frame.to_dict(), handle, indent=2, ensure_ascii=False)
    return job_dir


def _write_xyz(path: Path, species: list[str], positions: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(species)}\n")
        handle.write("active_learning_candidate\n")
        for element, xyz in zip(species, positions):
            handle.write(f"{element:2s} {xyz[0]:18.10f} {xyz[1]:18.10f} {xyz[2]:18.10f}\n")


def _command_tokens(command: str, nproc: int, input_path: Path) -> list[str]:
    tokens = shlex.split(command)
    if not tokens:
        raise ValueError("CP2K command cannot be empty")
    launchers = {Path(token).name for token in tokens}
    if nproc > 1 and launchers.isdisjoint({"mpirun", "srun"}):
        tokens = ["mpirun", "-np", str(nproc), *tokens]
    if "-i" not in tokens and str(input_path.name) not in tokens:
        tokens.extend(["-i", input_path.name])
    if "-o" not in tokens and "cp2k.out" not in tokens:
        tokens.extend(["-o", "cp2k.out"])
    return tokens


def _render_cp2k_input(
    template_text: str,
    species: list[str],
    positions: np.ndarray,
    cell: np.ndarray,
    project_name: str,
) -> str:
    geometry_lines = "\n".join(
        f"      {element:2s} {xyz[0]:18.10f} {xyz[1]:18.10f} {xyz[2]:18.10f}"
        for element, xyz in zip(species, positions)
    )
    cell_lines = "\n".join(
        [
            f"      A {cell[0, 0]:18.10f} {cell[0, 1]:18.10f} {cell[0, 2]:18.10f}",
            f"      B {cell[1, 0]:18.10f} {cell[1, 1]:18.10f} {cell[1, 2]:18.10f}",
            f"      C {cell[2, 0]:18.10f} {cell[2, 1]:18.10f} {cell[2, 2]:18.10f}",
            "      PERIODIC XYZ",
        ]
    )
    text = template_text
    text = re.sub(r"(?im)^(\s*PROJECT(?:_NAME)?\s+).*$", rf"\1{project_name}", text, count=1)
    if re.search(r"(?im)^\s*RUN_TYPE\s+", text):
        text = re.sub(r"(?im)^(\s*RUN_TYPE\s+).*$", r"\1ENERGY_FORCE", text, count=1)
    else:
        text = re.sub(r"(?im)^(\s*&END\s+GLOBAL\s*)$", "  RUN_TYPE ENERGY_FORCE\n\\1", text, count=1)
    text = _replace_section_body(text, "COORD", geometry_lines)
    text = _replace_section_body(text, "CELL", cell_lines)
    text = _ensure_force_print_block(text)
    return text


def _ensure_force_print_block(text: str) -> str:
    if re.search(r"(?im)^\s*FILENAME\s+active_forces\s*$", text):
        return text
    block = (
        "  &PRINT\n"
        "    &FORCES ON\n"
        "      FILENAME active_forces\n"
        "    &END FORCES\n"
        "  &END PRINT\n"
    )
    return re.sub(r"(?im)^(\s*&END\s+FORCE_EVAL\s*)$", block + r"\1", text, count=1)


def _replace_section_body(text: str, section: str, body: str) -> str:
    pattern = re.compile(
        rf"(?ims)(^[ \t]*&{section}\b.*?$)(.*?)(^[ \t]*&END\s+{section}\b.*?$)",
        flags=re.MULTILINE | re.DOTALL,
    )

    def repl(match: re.Match[str]) -> str:
        start = match.group(1)
        end = match.group(3)
        return f"{start}\n{body}\n{end}"

    new_text, count = pattern.subn(repl, text, count=1)
    if count == 0:
        raise ValueError(f"Could not find &{section} ... &END {section} block in CP2K template")
    return new_text


def _parse_cp2k_outputs(job_dir: Path, natoms: int) -> tuple[float, np.ndarray]:
    combined = []
    for path in sorted(job_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name in {"input.inp", "structure.xyz", "candidate_metadata.json"}:
            continue
        if path.exists():
            combined.append(path.read_text(encoding="utf-8", errors="replace"))
    text = "\n".join(combined)
    energy = _parse_energy(text)
    forces_au = _parse_forces(text, natoms=natoms)
    return energy, forces_au * FORCE_AU_TO_HARTREE_PER_ANGSTROM


def _parse_energy(text: str) -> float:
    patterns = [
        r"ENERGY\|\s+Total FORCE_EVAL.*?([-+]?\d+(?:\.\d+)?(?:[EeDd][-+]?\d+)?)",
        r"Total energy:\s*([-+]?\d+(?:\.\d+)?(?:[EeDd][-+]?\d+)?)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.MULTILINE)
        if matches:
            return float(matches[-1].replace("D", "e").replace("d", "e"))
    raise ValueError("Could not parse CP2K total energy")


def _parse_forces(text: str, natoms: int) -> np.ndarray:
    block_pattern = re.compile(
        r"ATOMIC FORCES.*?"
        r"(?:\n|\r\n)+\s*#?\s*Atom\s+Kind\s+Element\s+X\s+Y\s+Z\s*"
        r"(.*?)"
        r"\s*SUM OF ATOMIC FORCES",
        flags=re.DOTALL,
    )
    matches = block_pattern.findall(text)
    if matches:
        rows = _parse_force_rows(matches[-1].splitlines(), natoms=natoms)
        if rows is not None:
            return rows

    header_pattern = re.compile(r"^\s*#?\s*Atom\s+Kind\s+Element\s+X\s+Y\s+Z\s*$", flags=re.IGNORECASE)
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if not header_pattern.match(line):
            continue
        rows = _parse_force_rows(lines[idx + 1 :], natoms=natoms)
        if rows is not None:
            return rows
    raise ValueError("Could not parse CP2K atomic forces block")


def _parse_force_rows(lines: list[str], natoms: int) -> np.ndarray | None:
    rows: list[list[float]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if rows:
                break
            continue
        parts = stripped.split()
        if len(parts) < 6:
            if rows:
                break
            continue
        try:
            int(parts[0])
        except ValueError:
            if rows:
                break
            continue
        try:
            rows.append(
                [
                    float(parts[-3].replace("D", "e").replace("d", "e")),
                    float(parts[-2].replace("D", "e").replace("d", "e")),
                    float(parts[-1].replace("D", "e").replace("d", "e")),
                ]
            )
        except ValueError:
            if rows:
                break
            continue
        if len(rows) == natoms:
            return np.asarray(rows, dtype=np.float64)
    return None
