"""Analyze distance coverage and diversity of representative descriptors."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from descriptor_analysis_common import (
    DEFAULT_FEATURES,
    DEFAULT_REPRESENTATIVES,
    DEFAULT_REPORT_DIR,
    cdist_minibatch,
    configure_matplotlib,
    dump_csv,
    dump_json,
    load_pickle,
    pairwise_distances,
    percentile_summary,
    representative_rows,
    representative_source_indices,
    standardize,
)


configure_matplotlib()
import matplotlib.pyplot as plt  # noqa: E402


def plot_coverage_hist(selected_min: np.ndarray, random_means: np.ndarray, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), dpi=180)
    ax.hist(selected_min, bins=60, alpha=0.75, color="#4c78a8", label="selected representatives")
    if random_means.size:
        ax.axvline(random_means.mean(), color="#f58518", linestyle="--", label="random mean coverage")
    ax.set_xlabel("Distance to nearest representative")
    ax.set_ylabel("Sample count")
    ax.set_title("Feature-space coverage by representative structures")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_pairwise(pairwise: np.ndarray, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5), dpi=180)
    image = ax.imshow(pairwise, cmap="magma")
    fig.colorbar(image, ax=ax, label="Standardized Euclidean distance")
    ax.set_xlabel("Representative rank")
    ax.set_ylabel("Representative rank")
    ax.set_title("Representative pairwise distances")
    ranks = np.arange(pairwise.shape[0])
    ax.set_xticks(ranks)
    ax.set_yticks(ranks)
    ax.set_xticklabels(ranks + 1, rotation=90, fontsize=7)
    ax.set_yticklabels(ranks + 1, fontsize=7)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_rank_curve(rows: list[dict], selected_scaled: np.ndarray, output: Path) -> list[dict]:
    rank_rows: list[dict] = []
    for idx in range(len(selected_scaled)):
        if idx == 0:
            nearest = np.nan
        else:
            nearest = float(np.min(np.linalg.norm(selected_scaled[idx] - selected_scaled[:idx], axis=1)))
        row = rows[idx] if idx < len(rows) else {}
        rank_rows.append(
            {
                "rank": int(row.get("rank", idx + 1)),
                "source_record_index": int(row.get("source_record_index", -1)),
                "mol_id": int(row.get("mol_id", -1)),
                "traj": str(row.get("traj", "")),
                "frame": int(row.get("frame", -1)),
                "nearest_previous_distance": nearest,
            }
        )
    fig, ax = plt.subplots(figsize=(7, 4), dpi=180)
    ranks = [row["rank"] for row in rank_rows]
    distances = [row["nearest_previous_distance"] for row in rank_rows]
    ax.plot(ranks[1:], distances[1:], marker="o", color="#4c78a8")
    ax.set_xlabel("Representative rank")
    ax.set_ylabel("Distance to nearest earlier representative")
    ax.set_title("Incremental diversity during farthest-point selection")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return rank_rows


def random_baseline(
    scaled: np.ndarray,
    k: int,
    trials: int,
    batch_size: int,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for trial in range(trials):
        choice = rng.choice(len(scaled), size=k, replace=False)
        distances = cdist_minibatch(scaled, scaled[choice], batch_size=batch_size)
        nearest = distances.min(axis=1)
        pw = pairwise_distances(scaled[choice])
        upper = pw[np.triu_indices(k, k=1)]
        rows.append(
            {
                "trial": trial,
                "coverage_mean": float(nearest.mean()),
                "coverage_p95": float(np.percentile(nearest, 95)),
                "coverage_max": float(nearest.max()),
                "pairwise_mean": float(upper.mean()) if upper.size else 0.0,
                "pairwise_min": float(upper.min()) if upper.size else 0.0,
            }
        )
    return rows


def plot_random_baseline(selected_summary: dict, random_rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=180)
    metrics = [
        ("coverage_mean", "Mean nearest-representative distance", "lower is better"),
        ("pairwise_mean", "Mean representative pairwise distance", "higher is better"),
    ]
    for ax, (metric, title, subtitle) in zip(axes, metrics):
        values = np.asarray([row[metric] for row in random_rows], dtype=np.float64)
        ax.hist(values, bins=25, color="#bab0ab", alpha=0.8, label="random")
        ax.axvline(selected_summary[metric], color="#d62728", linewidth=2, label="selected")
        ax.set_title(f"{title}\n{subtitle}")
        ax.set_xlabel(metric)
        ax.set_ylabel("Trials")
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--representatives", type=Path, default=DEFAULT_REPRESENTATIVES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR / "distances")
    parser.add_argument("--distance-batch-size", type=int, default=4096)
    parser.add_argument("--random-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_data = load_pickle(args.features)
    representative_data = load_pickle(args.representatives)
    features = np.asarray(feature_data["features"], dtype=np.float32)
    scaled, mean, std = standardize(features)
    selected = representative_source_indices(representative_data)
    rows = representative_rows(representative_data, selected)
    selected_scaled = scaled[selected]
    distances = cdist_minibatch(scaled, selected_scaled, batch_size=args.distance_batch_size)
    nearest = distances.min(axis=1)
    nearest_rank = distances.argmin(axis=1) + 1
    pairwise = pairwise_distances(selected_scaled)
    upper = pairwise[np.triu_indices(len(selected), k=1)]
    rank_rows = plot_rank_curve(rows, selected_scaled, args.output_dir / "rank_distance_curve.png")
    random_rows = random_baseline(
        scaled,
        k=len(selected),
        trials=args.random_trials,
        batch_size=args.distance_batch_size,
        seed=args.seed,
    )
    selected_summary = {
        "coverage_mean": float(nearest.mean()),
        "coverage_p95": float(np.percentile(nearest, 95)),
        "coverage_max": float(nearest.max()),
        "pairwise_mean": float(upper.mean()) if upper.size else 0.0,
        "pairwise_min": float(upper.min()) if upper.size else 0.0,
    }

    plot_coverage_hist(nearest, np.asarray([row["coverage_mean"] for row in random_rows]), args.output_dir / "coverage_distance_hist.png")
    plot_pairwise(pairwise, args.output_dir / "representative_pairwise_distance.png")
    plot_random_baseline(selected_summary, random_rows, args.output_dir / "random_baseline_comparison.png")

    coverage_rows = [
        {
            "source_record_index": idx,
            "nearest_representative_rank": int(nearest_rank[idx]),
            "nearest_representative_distance": float(nearest[idx]),
            "mol_id": int(feature_data["mol_ids"][idx]),
            "traj": str(feature_data["traj"][idx]),
            "frame": int(feature_data["frame"][idx]),
        }
        for idx in range(len(features))
    ]
    dump_csv(args.output_dir / "coverage_distances.csv", coverage_rows)
    dump_csv(args.output_dir / "rank_distance_curve.csv", rank_rows)
    dump_csv(args.output_dir / "random_baseline.csv", random_rows)
    summary = {
        "features": str(args.features),
        "representatives": str(args.representatives),
        "records": int(len(features)),
        "representative_count": int(len(selected)),
        "distance": "standardized_euclidean",
        "coverage": percentile_summary(nearest),
        "representative_pairwise": percentile_summary(upper) if upper.size else {},
        "selected_summary": selected_summary,
        "random_trials": int(args.random_trials),
        "random_summary": {
            "coverage_mean": percentile_summary(np.asarray([row["coverage_mean"] for row in random_rows])),
            "coverage_p95": percentile_summary(np.asarray([row["coverage_p95"] for row in random_rows])),
            "pairwise_mean": percentile_summary(np.asarray([row["pairwise_mean"] for row in random_rows])),
        },
        "feature_mean": mean,
        "feature_std": std,
    }
    dump_json(args.output_dir / "distance_summary.json", summary)
    with (args.output_dir / "analysis.pkl").open("wb") as handle:
        pickle.dump(
            {
                "nearest_distance": nearest,
                "nearest_rank": nearest_rank,
                "pairwise_representative_distance": pairwise,
                "random_baseline": random_rows,
                "summary": summary,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(
        f"wrote_distance_analysis={args.output_dir} records={len(features)} "
        f"coverage_mean={selected_summary['coverage_mean']:.4f} pairwise_mean={selected_summary['pairwise_mean']:.4f}"
    )


if __name__ == "__main__":
    main()
