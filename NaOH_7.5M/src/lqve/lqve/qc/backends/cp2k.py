"""CP2K command-line energy backend."""

from __future__ import annotations

from pathlib import Path
import shlex

import numpy as np

from .external import ExternalCommandBackend, parse_with_patterns, render_template, xyz_geometry_lines


class CP2KBackend(ExternalCommandBackend):
    name = "cp2k"
    input_suffix = ".inp"

    def __init__(
        self,
        command: str = "cp2k.popt",
        work_dir: str | Path = "cp2k_work",
        nproc: int = 1,
        **options: object,
    ) -> None:
        super().__init__(command=command, work_dir=work_dir, nproc=nproc, **options)
        self.nproc = int(nproc)

    def prepare_input(
        self,
        job_dir: Path,
        species: list[str],
        positions: np.ndarray,
        grid_point: np.ndarray,
    ) -> Path:
        geometry_path = job_dir / "geometry.xyz"
        self._write_geometry_xyz(geometry_path, species, positions, grid_point)
        input_path = job_dir / "input.inp"
        geometry = xyz_geometry_lines(species, positions)
        if self.template is None:
            text = (
                "&GLOBAL\n"
                "  PROJECT lqve\n"
                "  RUN_TYPE ENERGY\n"
                "&END GLOBAL\n"
                "&FORCE_EVAL\n"
                "  METHOD Quickstep\n"
                "  &SUBSYS\n"
                "    &COORD\n"
                f"{geometry}\n"
                "    &END COORD\n"
                "  &END SUBSYS\n"
                "&END FORCE_EVAL\n"
            )
        else:
            text = render_template(self.template, geometry, geometry_path.name)
        input_path.write_text(text, encoding="utf-8")
        return input_path

    def _write_geometry_xyz(
        self,
        path: Path,
        species: list[str],
        positions: np.ndarray,
        grid_point: np.ndarray,
    ) -> None:
        from lqve.qc.io import write_xyz

        write_xyz(path, species, positions, comment=f"grid={grid_point.tolist()}")

    def command_for_input(self, input_path: Path) -> list[str]:
        command = shlex.split(self.command)
        if self.nproc > 1 and (not command or "mpirun" not in command[0]):
            command = ["mpirun", "-np", str(self.nproc), *command]
        return [*command, input_path.name]

    def parse_energy(self, output_text: str, job_dir: Path) -> float:
        combined = output_text
        for output_file in list(job_dir.glob("*.out")) + list(job_dir.glob("*.log")):
            combined += "\n" + output_file.read_text(encoding="utf-8", errors="replace")
        return parse_with_patterns(
            combined,
            [
                r"Total energy:\s*([-+]?\d+(?:\.\d+)?(?:[EeDd][-+]?\d+)?)",
                r"ENERGY\|.*?([-+]?\d+(?:\.\d+)?(?:[EeDd][-+]?\d+)?)",
            ],
        )
