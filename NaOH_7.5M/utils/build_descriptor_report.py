"""Build a complete descriptor-analysis report."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from descriptor_analysis_common import DEFAULT_FEATURES, DEFAULT_REPRESENTATIVES, DEFAULT_REPORT_DIR


def run_step(command: list[str]) -> None:
    print("running:", " ".join(command))
    subprocess.run(command, check=True)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv_rows(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def rel(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def write_report(output_dir: Path, features: Path, representatives: Path) -> None:
    pca_dir = output_dir / "pca"
    distance_dir = output_dir / "distances"
    structure_dir = output_dir / "structures"
    pca_summary = load_json(pca_dir / "pca_summary.json")
    distance_summary = load_json(distance_dir / "distance_summary.json")
    structure_rows = load_csv_rows(structure_dir / "structure_stats.csv")
    first_rows = structure_rows[:20]

    lines: list[str] = []
    lines.append("# ViSNet Descriptor Representative Analysis Report")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Features: `{features}`")
    lines.append(f"- Representatives: `{representatives}`")
    lines.append(f"- Records: `{pca_summary.get('records', 'unknown')}`")
    lines.append(f"- Representative count: `{distance_summary.get('representative_count', 'unknown')}`")
    lines.append("")
    lines.append("## PCA Distribution")
    lines.append("")
    lines.append(f"![PCA 2D]({rel(pca_dir / 'pca_2d.png', output_dir)})")
    lines.append("")
    lines.append(f"![PCA 3D]({rel(pca_dir / 'pca_3d.png', output_dir)})")
    lines.append("")
    lines.append(f"![PCA variance]({rel(pca_dir / 'pca_variance.png', output_dir)})")
    lines.append("")
    if pca_summary:
        variance = pca_summary.get("explained_variance_ratio", [])
        lines.append(f"Explained variance ratio for PC1-3: `{variance}`.")
        lines.append("")
    lines.append("## Feature-Space Coverage")
    lines.append("")
    coverage = distance_summary.get("coverage", {})
    pairwise = distance_summary.get("representative_pairwise", {})
    selected = distance_summary.get("selected_summary", {})
    lines.append(f"- Mean distance from all samples to nearest representative: `{selected.get('coverage_mean', 'unknown')}`")
    lines.append(f"- 95th percentile nearest-representative distance: `{coverage.get('p95', 'unknown')}`")
    lines.append(f"- Mean pairwise distance among representatives: `{selected.get('pairwise_mean', 'unknown')}`")
    lines.append(f"- Minimum pairwise distance among representatives: `{pairwise.get('min', 'unknown')}`")
    lines.append("")
    lines.append(f"![Coverage histogram]({rel(distance_dir / 'coverage_distance_hist.png', output_dir)})")
    lines.append("")
    lines.append(f"![Pairwise distances]({rel(distance_dir / 'representative_pairwise_distance.png', output_dir)})")
    lines.append("")
    lines.append(f"![Rank distance curve]({rel(distance_dir / 'rank_distance_curve.png', output_dir)})")
    lines.append("")
    lines.append(f"![Random baseline comparison]({rel(distance_dir / 'random_baseline_comparison.png', output_dir)})")
    lines.append("")
    lines.append("## Representative Geometry")
    lines.append("")
    lines.append(f"![Bond length histogram]({rel(structure_dir / 'bond_length_hist.png', output_dir)})")
    lines.append("")
    lines.append(f"![Angle histogram]({rel(structure_dir / 'angle_hist.png', output_dir)})")
    lines.append("")
    nearest_na = structure_dir / "nearest_Na_distance_hist.png"
    if nearest_na.exists():
        lines.append(f"![Nearest Na distance]({rel(nearest_na, output_dir)})")
        lines.append("")
    lines.append("## Representative List")
    lines.append("")
    if first_rows:
        columns = ["rank", "mol_id", "traj", "frame", "molecule_pattern", "primary_bond_1", "primary_bond_2", "primary_angle_deg", "structure_image"]
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for row in first_rows:
            values = []
            for column in columns:
                value = row.get(column, "")
                if column == "structure_image" and value:
                    value = f"[{value}]({rel(structure_dir / value, output_dir)})"
                values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
    else:
        lines.append("No structure rows were found.")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- PCA plots test whether selected representatives occupy boundaries or sparse regions of the learned descriptor space.")
    lines.append("- Coverage distances test whether all samples are close to at least one representative.")
    lines.append("- Pairwise representative distances test whether selected structures are redundant.")
    lines.append("- Geometry statistics test whether descriptor-space differences correspond to interpretable structural changes.")
    lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--representatives", type=Path, default=DEFAULT_REPRESENTATIVES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--distance-batch-size", type=int, default=4096)
    parser.add_argument("--random-trials", type=int, default=100)
    parser.add_argument("--max-background-points", type=int, default=20000)
    parser.add_argument("--coordination-cutoff", type=float, default=3.2)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    script_dir = Path(__file__).resolve().parent
    run_step(
        [
            sys.executable,
            str(script_dir / "plot_descriptor_pca.py"),
            "--features",
            str(args.features),
            "--representatives",
            str(args.representatives),
            "--output-dir",
            str(args.output_dir / "pca"),
            "--max-background-points",
            str(args.max_background_points),
        ]
    )
    run_step(
        [
            sys.executable,
            str(script_dir / "analyze_representatives.py"),
            "--features",
            str(args.features),
            "--representatives",
            str(args.representatives),
            "--output-dir",
            str(args.output_dir / "distances"),
            "--distance-batch-size",
            str(args.distance_batch_size),
            "--random-trials",
            str(args.random_trials),
        ]
    )
    run_step(
        [
            sys.executable,
            str(script_dir / "plot_representative_structures.py"),
            "--features",
            str(args.features),
            "--representatives",
            str(args.representatives),
            "--output-dir",
            str(args.output_dir / "structures"),
            "--coordination-cutoff",
            str(args.coordination_cutoff),
        ]
    )
    write_report(args.output_dir, args.features, args.representatives)
    print(f"wrote_descriptor_report={args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
