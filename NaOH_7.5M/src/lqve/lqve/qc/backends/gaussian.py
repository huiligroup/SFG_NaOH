"""Gaussian command-line energy backend."""

from __future__ import annotations

from pathlib import Path
import shlex

import numpy as np

from .external import ExternalCommandBackend, parse_with_patterns, render_template, xyz_geometry_lines


class GaussianBackend(ExternalCommandBackend):
    name = "gaussian"
    input_suffix = ".gjf"

    def __init__(self, command: str = "g16", work_dir: str | Path = "gaussian_work", **options: object) -> None:
        super().__init__(command=command, work_dir=work_dir, **options)

    def prepare_input(
        self,
        job_dir: Path,
        species: list[str],
        positions: np.ndarray,
        grid_point: np.ndarray,
    ) -> Path:
        input_path = job_dir / "input.gjf"
        geometry = xyz_geometry_lines(species, positions)
        if self.template is None:
            text = (
                "%chk=lqve.chk\n"
                "#p b3lyp/6-31g(d) scf=tight\n\n"
                f"LQVE grid {grid_point.tolist()}\n\n"
                "1 1\n"
                f"{geometry}\n\n"
            )
        else:
            text = render_template(self.template, geometry, "geometry.xyz")
        input_path.write_text(text, encoding="utf-8")
        return input_path

    def command_for_input(self, input_path: Path) -> list[str]:
        return [*shlex.split(self.command), input_path.name]

    def parse_energy(self, output_text: str, job_dir: Path) -> float:
        combined = output_text
        for output_file in list(job_dir.glob("*.log")) + list(job_dir.glob("*.out")):
            combined += "\n" + output_file.read_text(encoding="utf-8", errors="replace")
        return parse_with_patterns(
            combined,
            [
                r"SCF Done:\s+E\([^)]+\)\s+=\s+([-+]?\d+(?:\.\d+)?(?:[EeDd][-+]?\d+)?)",
                r"E\([^)]+\)\s*=\s*([-+]?\d+(?:\.\d+)?(?:[EeDd][-+]?\d+)?)",
            ],
        )
