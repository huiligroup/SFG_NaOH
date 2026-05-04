"""Plot and summarize representative molecule structures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from descriptor_analysis_common import (
    DEFAULT_FEATURES,
    DEFAULT_REPRESENTATIVES,
    DEFAULT_REPORT_DIR,
    ELEMENT_COLORS,
    configure_matplotlib,
    dump_csv,
    dump_json,
    load_pickle,
    pbc_angle,
    pbc_distance,
    pbc_distances,
    record_frame_position,
    representative_rows,
    representative_source_indices,
)


configure_matplotlib()
import matplotlib.pyplot as plt  # noqa: E402


def internal_geometry(mol_pos: np.ndarray, mol_species: list[str], cell: np.ndarray) -> dict[str, float | str]:
    row: dict[str, float | str] = {"molecule_pattern": "-".join(mol_species)}
    if len(mol_pos) >= 2:
        row["d_0_1"] = pbc_distance(mol_pos[0], mol_pos[1], cell)
    if len(mol_pos) >= 3:
        row["d_0_2"] = pbc_distance(mol_pos[0], mol_pos[2], cell)
        row["d_1_2"] = pbc_distance(mol_pos[1], mol_pos[2], cell)
        if mol_species == ["O", "H", "H"]:
            row["primary_bond_1"] = row["d_0_1"]
            row["primary_bond_2"] = row["d_0_2"]
            row["primary_angle_deg"] = pbc_angle(mol_pos[1], mol_pos[0], mol_pos[2], cell)
            row["primary_angle_label"] = "H-O-H"
        elif mol_species == ["Na", "O", "H"]:
            row["primary_bond_1"] = row["d_0_1"]
            row["primary_bond_2"] = row["d_1_2"]
            row["primary_angle_deg"] = pbc_angle(mol_pos[0], mol_pos[1], mol_pos[2], cell)
            row["primary_angle_label"] = "Na-O-H"
        else:
            row["primary_bond_1"] = row["d_0_1"]
            row["primary_bond_2"] = row["d_1_2"]
            row["primary_angle_deg"] = pbc_angle(mol_pos[0], mol_pos[1], mol_pos[2], cell)
            row["primary_angle_label"] = "0-1-2"
    return row


def environment_stats(
    full_pos: np.ndarray,
    species: np.ndarray,
    mol_pos: np.ndarray,
    atom_indices: list[int],
    cell: np.ndarray,
    coordination_cutoff: float,
) -> dict[str, float | int]:
    selected = set(atom_indices)
    row: dict[str, float | int] = {}
    for element in ("Na", "O", "H"):
        candidate_idx = [idx for idx, symbol in enumerate(species) if str(symbol) == element and idx not in selected]
        if not candidate_idx:
            row[f"nearest_{element}_distance"] = float("nan")
            row[f"coordination_{element}_{coordination_cutoff:.2f}A"] = 0
            continue
        distances = pbc_distances(full_pos[candidate_idx], mol_pos, cell)
        min_by_atom = distances.min(axis=1)
        row[f"nearest_{element}_distance"] = float(min_by_atom.min())
        row[f"coordination_{element}_{coordination_cutoff:.2f}A"] = int(np.sum(min_by_atom <= coordination_cutoff))
    return row


def set_axes_equal(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def plot_structure(
    full_pos: np.ndarray,
    species: np.ndarray,
    mol_pos: np.ndarray,
    atom_indices: list[int],
    mol_species: list[str],
    row: dict,
    output: Path,
) -> None:
    fig = plt.figure(figsize=(7, 6), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    selected = set(atom_indices)
    for element in ("H", "O", "Na"):
        idx = [i for i, symbol in enumerate(species) if str(symbol) == element and i not in selected]
        if idx:
            xyz = full_pos[idx]
            ax.scatter(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                s=8 if element != "H" else 4,
                c=ELEMENT_COLORS.get(element, "#999999"),
                alpha=0.16,
                linewidths=0,
                label=f"{element} background",
            )
    sizes = [95 if element != "H" else 55 for element in mol_species]
    colors = [ELEMENT_COLORS.get(element, "#222222") for element in mol_species]
    ax.scatter(
        mol_pos[:, 0],
        mol_pos[:, 1],
        mol_pos[:, 2],
        s=sizes,
        c=colors,
        edgecolors="black",
        linewidths=0.8,
        label="representative molecule",
    )
    for i in range(len(mol_pos)):
        ax.text(mol_pos[i, 0], mol_pos[i, 1], mol_pos[i, 2], f"{mol_species[i]}{atom_indices[i]}", fontsize=8)
    if len(mol_pos) >= 2:
        ax.plot(mol_pos[:2, 0], mol_pos[:2, 1], mol_pos[:2, 2], color="black", linewidth=1.5)
    if len(mol_pos) >= 3:
        ax.plot(mol_pos[[0, 2], 0], mol_pos[[0, 2], 1], mol_pos[[0, 2], 2], color="black", linewidth=1.0, linestyle="--")
        ax.plot(mol_pos[1:, 0], mol_pos[1:, 1], mol_pos[1:, 2], color="black", linewidth=1.0, linestyle=":")
    ax.set_title(f"rank {row['rank']} mol {row['mol_id']} traj {row['traj']} frame {row['frame']}")
    ax.set_xlabel("x (A)")
    ax.set_ylabel("y (A)")
    ax.set_zlabel("z (A)")
    set_axes_equal(ax, full_pos)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def hist_plot(rows: list[dict], key: str, output: Path, xlabel: str) -> None:
    values = np.asarray([row[key] for row in rows if key in row and np.isfinite(float(row[key]))], dtype=np.float64)
    if values.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4), dpi=180)
    ax.hist(values, bins=min(20, max(5, values.size)), color="#4c78a8", alpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Representative count")
    ax.set_title(xlabel)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--representatives", type=Path, default=DEFAULT_REPRESENTATIVES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR / "structures")
    parser.add_argument("--coordination-cutoff", type=float, default=3.2)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_data = load_pickle(args.features)
    representative_data = load_pickle(args.representatives)
    selected = representative_source_indices(representative_data)
    rows = representative_rows(representative_data, selected)
    species = np.asarray(feature_data["species"], dtype=object)
    stats_rows: list[dict] = []

    for fallback_rank, (source_idx, rep_row) in enumerate(zip(selected, rows), start=1):
        rank = int(rep_row.get("rank", fallback_rank))
        atom_indices = np.asarray(feature_data["atom_indices"][source_idx], dtype=np.int64).tolist()
        mol_species = [str(item) for item in np.asarray(feature_data["mol_species"][source_idx], dtype=object).tolist()]
        mol_pos = np.asarray(feature_data["pos"][source_idx], dtype=np.float32)
        full_pos, cell = record_frame_position(feature_data, int(source_idx))
        row = {
            "rank": rank,
            "source_record_index": int(source_idx),
            "mol_id": int(feature_data["mol_ids"][source_idx]),
            "traj": str(feature_data["traj"][source_idx]),
            "frame": int(feature_data["frame"][source_idx]),
            "frame_index": int(feature_data["frame_index"][source_idx]),
            "atom_indices": " ".join(map(str, atom_indices)),
            "mol_species": " ".join(mol_species),
        }
        row.update(internal_geometry(mol_pos, mol_species, cell))
        row.update(environment_stats(full_pos, species, mol_pos, atom_indices, cell, args.coordination_cutoff))
        image_name = f"rank_{rank:04d}_structure_3d.png"
        row["structure_image"] = image_name
        plot_structure(full_pos, species, mol_pos, atom_indices, mol_species, row, args.output_dir / image_name)
        stats_rows.append(row)

    dump_csv(args.output_dir / "structure_stats.csv", stats_rows)
    hist_plot(stats_rows, "primary_bond_1", args.output_dir / "bond_length_hist.png", "Primary bond length 1 (A)")
    hist_plot(stats_rows, "primary_bond_2", args.output_dir / "bond_length_2_hist.png", "Primary bond length 2 (A)")
    hist_plot(stats_rows, "primary_angle_deg", args.output_dir / "angle_hist.png", "Primary angle (deg)")
    for element in ("Na", "O", "H"):
        hist_plot(stats_rows, f"nearest_{element}_distance", args.output_dir / f"nearest_{element}_distance_hist.png", f"Nearest {element} distance (A)")
        hist_plot(
            stats_rows,
            f"coordination_{element}_{args.coordination_cutoff:.2f}A",
            args.output_dir / f"coordination_{element}_hist.png",
            f"{element} coordination within {args.coordination_cutoff:.2f} A",
        )
    dump_json(
        args.output_dir / "structure_summary.json",
        {
            "features": str(args.features),
            "representatives": str(args.representatives),
            "representative_count": len(stats_rows),
            "coordination_cutoff": args.coordination_cutoff,
            "outputs": ["structure_stats.csv", "bond_length_hist.png", "angle_hist.png", "rank_XXXX_structure_3d.png"],
        },
    )
    print(f"wrote_structure_analysis={args.output_dir} representatives={len(stats_rows)}")


if __name__ == "__main__":
    main()
