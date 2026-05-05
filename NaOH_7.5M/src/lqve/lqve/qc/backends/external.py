"""Command-line quantum chemistry backend helpers."""

from __future__ import annotations

import re
import shutil
import shlex
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .base import EnergyBackend
from lqve.qc.io import write_xyz
from lqve.qc.types import EnergyResult, GeometryBatch


class ExternalCommandBackend(EnergyBackend):
    """Base implementation for one-command-per-geometry QC backends."""

    input_suffix = ".xyz"
    output_name = "stdout.log"

    def __init__(
        self,
        command: str,
        work_dir: str | Path,
        template: str | Path | None = None,
        keep_workdirs: bool = False,
        **options: object,
    ) -> None:
        super().__init__(
            command=command,
            work_dir=str(work_dir),
            template=None if template is None else str(template),
            keep_workdirs=keep_workdirs,
            **options,
        )
        self.command = command
        self.work_dir = Path(work_dir)
        self.template = None if template is None else Path(template)
        self.keep_workdirs = keep_workdirs

    def evaluate(self, batch: GeometryBatch) -> EnergyResult:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        energies = np.full(batch.geometry_count, np.nan, dtype=np.float64)
        success = np.zeros(batch.geometry_count, dtype=bool)
        errors: list[str] = []
        manifest: list[dict[str, object]] = []
        start = time.perf_counter()
        for local_idx, positions in enumerate(batch.positions):
            source_idx = int(batch.source_indices[local_idx]) if batch.source_indices is not None else local_idx
            job_dir = self._unique_job_dir(batch.frame_id, source_idx)
            job_dir.mkdir(parents=True, exist_ok=False)
            try:
                input_path = self.prepare_input(job_dir, batch.species, positions, batch.grid_points[local_idx])
                completed = self.run_command(job_dir, input_path)
                output_text = completed.stdout + "\n" + completed.stderr
                (job_dir / self.output_name).write_text(output_text, encoding="utf-8")
                energies[local_idx] = self.parse_energy(output_text, job_dir)
                success[local_idx] = np.isfinite(energies[local_idx])
                manifest.append(
                    {
                        "frame_id": batch.frame_id,
                        "grid_index": source_idx,
                        "success": bool(success[local_idx]),
                        "energy_hartree": float(energies[local_idx]),
                        "work_dir": str(job_dir),
                    }
                )
                if success[local_idx] and not self.keep_workdirs:
                    shutil.rmtree(job_dir, ignore_errors=True)
            except Exception as exc:  # noqa: BLE001 - preserve per-geometry failures in result.
                message = f"frame={batch.frame_id} grid={source_idx}: {exc}"
                errors.append(message)
                manifest.append(
                    {
                        "frame_id": batch.frame_id,
                        "grid_index": source_idx,
                        "success": False,
                        "error": str(exc),
                        "work_dir": str(job_dir),
                    }
                )
        return EnergyResult(
            backend=self.name,
            frame_id=batch.frame_id,
            energies_hartree=energies,
            grid_points=batch.grid_points,
            success_mask=success,
            source_indices=batch.source_indices,
            errors=errors,
            metadata={
                "backend": self.name,
                "command": self.command,
                "work_dir": str(self.work_dir),
                "elapsed_seconds": time.perf_counter() - start,
                "manifest": manifest,
            },
        )

    def _unique_job_dir(self, frame_id: str, source_idx: int) -> Path:
        base = self.work_dir / f"{_safe_name(frame_id)}_grid_{source_idx:06d}"
        if not base.exists():
            return base
        suffix = 1
        while True:
            candidate = self.work_dir / f"{base.name}_run{suffix:03d}"
            if not candidate.exists():
                return candidate
            suffix += 1

    def prepare_input(
        self,
        job_dir: Path,
        species: list[str],
        positions: np.ndarray,
        grid_point: np.ndarray,
    ) -> Path:
        path = job_dir / f"geometry{self.input_suffix}"
        write_xyz(path, species, positions, comment=f"grid={grid_point.tolist()}")
        return path

    def run_command(self, job_dir: Path, input_path: Path) -> subprocess.CompletedProcess[str]:
        cmd = self.command_for_input(input_path)
        return subprocess.run(
            cmd,
            cwd=job_dir,
            text=True,
            capture_output=True,
            check=False,
        )

    def command_for_input(self, input_path: Path) -> list[str]:
        return [*shlex.split(self.command), input_path.name]

    def parse_energy(self, output_text: str, job_dir: Path) -> float:
        raise NotImplementedError


def parse_with_patterns(text: str, patterns: list[str], convert: Callable[[str], float] = float) -> float:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match:
            return convert(match.group(1).replace("D", "e").replace("d", "e"))
    raise ValueError("Could not parse energy from output")


def render_template(template: Path, geometry_lines: str, xyz_file: str) -> str:
    text = template.read_text(encoding="utf-8")
    if "{geometry}" in text or "{xyz_file}" in text:
        return text.format(geometry=geometry_lines, xyz_file=xyz_file)
    rendered: list[str] = []
    inserted = False
    for line in text.splitlines():
        rendered.append(line)
        if line.strip() in {"########", "1 1"}:
            rendered.extend(geometry_lines.splitlines())
            inserted = True
    if not inserted:
        rendered.append(geometry_lines)
    return "\n".join(rendered) + "\n"


def xyz_geometry_lines(species: list[str], positions: np.ndarray) -> str:
    return "\n".join(
        f"{element:2s} {xyz[0]:18.10f} {xyz[1]:18.10f} {xyz[2]:18.10f}"
        for element, xyz in zip(species, positions)
    )


def _safe_name(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value))
    return text or "frame"
