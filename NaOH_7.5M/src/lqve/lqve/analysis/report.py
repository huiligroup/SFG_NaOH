"""Markdown report generation for LQVE analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_report(
    path: str | Path,
    title: str,
    input_shift_dir: str,
    statistics: dict[str, Any],
    tables: dict[str, Path],
    figures: dict[str, Path],
    failed_frames: list[dict[str, Any]],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        "## Inputs",
        "",
        f"- Shift directory: `{input_shift_dir}`",
        "",
        "## Quality Summary",
        "",
        f"- Frames: {statistics.get('frame_count', 0)}",
        f"- Successful frames: {statistics.get('success_count', 0)}",
        f"- Failed frames: {statistics.get('failure_count', 0)}",
        f"- Success fraction: {statistics.get('success_fraction', 0.0):.4f}",
        f"- Transition NaN fraction: {statistics.get('transition_nan_fraction', 0.0):.4f}",
        f"- Shift NaN fraction: {statistics.get('shift_nan_fraction', 0.0):.4f}",
        f"- Transition count analyzed: {statistics.get('transition_count', 0)}",
        "",
        "## Input Integrity",
        "",
        f"- Reference mode: `{statistics.get('reference_mode', 'unknown')}`",
        f"- Selected transitions: `{statistics.get('selected_transitions', 'all')}`",
        f"- Summary CSV rows: {statistics.get('summary_rows', 0)}",
        f"- Reference usage: `{statistics.get('reference_counts', {})}`",
        "",
        "## Tables",
        "",
    ]
    for name, table_path in tables.items():
        lines.append(f"- {name}: `{_relative(path.parent, table_path)}`")
    lines.extend(["", "## Figures", ""])
    for name, figure_path in figures.items():
        rel = _relative(path.parent, figure_path)
        lines.append(f"### {name}")
        lines.append("")
        if figure_path.suffix.lower() == ".png":
            lines.append(f"![{name}]({rel})")
        else:
            lines.append(f"- `{rel}`")
        lines.append("")
    lines.extend(["## Failed Or Incomplete Frames", ""])
    if failed_frames:
        lines.append("| frame_index | frame_id | reference_id | source_qc_dir | transition_nan_count | shift_nan_count |")
        lines.append("| --- | --- | --- | --- | ---: | ---: |")
        for row in failed_frames[:50]:
            lines.append(
                "| {frame_index} | {frame_id} | {reference_id} | `{source_qc_dir}` | {transition_nan_count} | {shift_nan_count} |".format(
                    **row
                )
            )
        if len(failed_frames) > 50:
            lines.append("")
            lines.append(f"Only the first 50 of {len(failed_frames)} failed or incomplete frames are shown.")
    else:
        lines.append("No failed or incomplete frames were found.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _relative(base: Path, target: Path) -> str:
    try:
        return str(target.relative_to(base))
    except ValueError:
        return str(target)
