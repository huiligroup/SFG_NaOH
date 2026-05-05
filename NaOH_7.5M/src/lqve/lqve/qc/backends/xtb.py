"""xTB command-line energy backend."""

from __future__ import annotations

from pathlib import Path

from .external import ExternalCommandBackend, parse_with_patterns


class XTBBackend(ExternalCommandBackend):
    name = "xtb"

    def __init__(self, command: str = "xtb --gfn 2", work_dir: str | Path = "xtb_work", **options: object) -> None:
        super().__init__(command=command, work_dir=work_dir, **options)

    def parse_energy(self, output_text: str, job_dir: Path) -> float:
        combined = output_text
        for output_file in job_dir.glob("*.out"):
            combined += "\n" + output_file.read_text(encoding="utf-8", errors="replace")
        return parse_with_patterns(
            combined,
            [
                r"\|\s*TOTAL ENERGY\s+([-+]?\d+(?:\.\d+)?(?:[EeDd][-+]?\d+)?)\s+Eh\s*\|",
                r"TOTAL ENERGY\s+([-+]?\d+(?:\.\d+)?(?:[EeDd][-+]?\d+)?)",
            ],
        )
