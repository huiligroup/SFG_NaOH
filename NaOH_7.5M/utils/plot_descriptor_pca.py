"""Plot PCA views of ViSNet molecule descriptor vectors."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from descriptor_analysis_common import (
    DEFAULT_FEATURES,
    DEFAULT_REPRESENTATIVES,
    DEFAULT_REPORT_DIR,
    configure_matplotlib,
    dump_csv,
    dump_json,
    load_pickle,
    representative_rows,
    representative_source_indices,
    standardize,
)


configure_matplotlib()
import matplotlib.pyplot as plt  # noqa: E402


def background_indices(total: int, max_points: int, seed: int) -> np.ndarray:
    if total <= max_points:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=max_points, replace=False))


def annotate_representatives(ax, coords: np.ndarray, selected: np.ndarray, rows: list[dict]) -> None:
    for fallback_rank, (source_idx, row) in enumerate(zip(selected, rows), start=1):
        rank = int(row.get("rank", fallback_rank))
        ax.text(coords[source_idx, 0], coords[source_idx, 1], str(rank), fontsize=8, weight="bold")


def plot_2d(
    coords: np.ndarray,
    selected: np.ndarray,
    rows: list[dict],
    bg_idx: np.ndarray,
    output: Path,
    color_values: np.ndarray | None,
    color_label: str,
    title: str,
    explained: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)
    if color_values is None:
        ax.scatter(coords[bg_idx, 0], coords[bg_idx, 1], s=5, c="#b8b8b8", alpha=0.35, linewidths=0)
    else:
        scatter = ax.scatter(coords[bg_idx, 0], coords[bg_idx, 1], s=5, c=color_values[bg_idx], cmap="viridis", alpha=0.45, linewidths=0)
        fig.colorbar(scatter, ax=ax, label=color_label)
    ax.scatter(coords[selected, 0], coords[selected, 1], s=55, c="#d62728", edgecolors="black", linewidths=0.7, label="representatives")
    annotate_representatives(ax, coords, selected, rows)
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_3d(coords: np.ndarray, selected: np.ndarray, rows: list[dict], bg_idx: np.ndarray, output: Path, explained: np.ndarray) -> None:
    fig = plt.figure(figsize=(8, 6), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(coords[bg_idx, 0], coords[bg_idx, 1], coords[bg_idx, 2], s=4, c="#b8b8b8", alpha=0.25, linewidths=0)
    ax.scatter(
        coords[selected, 0],
        coords[selected, 1],
        coords[selected, 2],
        s=55,
        c="#d62728",
        edgecolors="black",
        linewidths=0.6,
    )
    for fallback_rank, (source_idx, row) in enumerate(zip(selected, rows), start=1):
        rank = int(row.get("rank", fallback_rank))
        ax.text(coords[source_idx, 0], coords[source_idx, 1], coords[source_idx, 2], str(rank), fontsize=8)
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
    ax.set_zlabel(f"PC3 ({explained[2] * 100:.1f}%)")
    ax.set_title("Descriptor PCA 3D")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_variance(explained: np.ndarray, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=180)
    pcs = np.arange(1, len(explained) + 1)
    ax.bar(pcs, explained * 100.0, color="#4c78a8")
    ax.plot(pcs, np.cumsum(explained) * 100.0, color="#f58518", marker="o", label="cumulative")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_xticks(pcs)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--representatives", type=Path, default=DEFAULT_REPRESENTATIVES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR / "pca")
    parser.add_argument("--max-background-points", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_data = load_pickle(args.features)
    representative_data = load_pickle(args.representatives)
    features = np.asarray(feature_data["features"], dtype=np.float32)
    scaled, _, _ = standardize(features)
    component_count = min(3, scaled.shape[0], scaled.shape[1])
    if component_count < 1:
        raise ValueError("PCA requires at least one descriptor record and one feature dimension")
    pca = PCA(n_components=component_count)
    fitted_coords = pca.fit_transform(scaled).astype(np.float32)
    coords = np.zeros((scaled.shape[0], 3), dtype=np.float32)
    coords[:, :component_count] = fitted_coords
    explained = np.zeros(3, dtype=np.float32)
    explained[:component_count] = pca.explained_variance_ratio_.astype(np.float32)

    selected = representative_source_indices(representative_data)
    rows = representative_rows(representative_data, selected)
    bg_idx = background_indices(len(features), args.max_background_points, args.seed)

    plot_2d(coords, selected, rows, bg_idx, args.output_dir / "pca_2d.png", None, "", "Descriptor PCA 2D", explained)
    plot_3d(coords, selected, rows, bg_idx, args.output_dir / "pca_3d.png", explained)
    plot_variance(explained, args.output_dir / "pca_variance.png")
    traj_codes, traj_names = np.unique(np.asarray(feature_data["traj"], dtype=str), return_inverse=True)
    plot_2d(
        coords,
        selected,
        rows,
        bg_idx,
        args.output_dir / "pca_by_traj.png",
        traj_names.astype(np.float32),
        "traj code",
        "Descriptor PCA colored by trajectory",
        explained,
    )
    plot_2d(
        coords,
        selected,
        rows,
        bg_idx,
        args.output_dir / "pca_by_mol_id.png",
        np.asarray(feature_data["mol_ids"], dtype=np.float32),
        "mol_id",
        "Descriptor PCA colored by molecule id",
        explained,
    )
    plot_2d(
        coords,
        selected,
        rows,
        bg_idx,
        args.output_dir / "pca_by_frame.png",
        np.asarray(feature_data["frame"], dtype=np.float32),
        "MD frame",
        "Descriptor PCA colored by frame",
        explained,
    )

    csv_rows = []
    selected_set = set(map(int, selected))
    rank_by_source = {int(row.get("source_record_index", idx)): int(row.get("rank", rank)) for rank, (idx, row) in enumerate(zip(selected, rows), start=1)}
    for idx in range(len(features)):
        csv_rows.append(
            {
                "source_record_index": idx,
                "pc1": float(coords[idx, 0]),
                "pc2": float(coords[idx, 1]),
                "pc3": float(coords[idx, 2]),
                "traj": str(feature_data["traj"][idx]),
                "frame": int(feature_data["frame"][idx]),
                "frame_index": int(feature_data["frame_index"][idx]),
                "mol_id": int(feature_data["mol_ids"][idx]),
                "is_representative": int(idx in selected_set),
                "representative_rank": rank_by_source.get(idx, ""),
            }
        )
    dump_csv(args.output_dir / "pca_coordinates.csv", csv_rows)
    dump_json(
        args.output_dir / "pca_summary.json",
        {
            "features": str(args.features),
            "representatives": str(args.representatives),
            "records": int(len(features)),
            "selected": selected.tolist(),
            "explained_variance_ratio": explained.tolist(),
            "traj_code_mapping": {str(name): int(code) for code, name in enumerate(traj_codes)},
            "outputs": [
                "pca_2d.png",
                "pca_3d.png",
                "pca_variance.png",
                "pca_by_traj.png",
                "pca_by_mol_id.png",
                "pca_by_frame.png",
                "pca_coordinates.csv",
            ],
        },
    )
    print(f"wrote_pca={args.output_dir} records={len(features)} pc1={explained[0]:.4f}")


if __name__ == "__main__":
    main()
