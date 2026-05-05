"""I/O helpers for LQVE energy calculations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from lqve.embedding.result import EmbeddingResult

from .types import EnergyResult, GeometryBatch


_Z_TABLE = {
    "H": 1,
    "D": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Na": 11,
    "Cl": 17,
}


def atomic_numbers(species: list[str]) -> np.ndarray:
    try:
        return np.asarray([_Z_TABLE[item] for item in species], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"No atomic number configured for element {exc.args[0]!r}") from exc


def parse_cell(value: str | list[float] | list[list[float]] | np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [float(item.strip()) for item in value.split(",") if item.strip()]
        if len(parts) not in {3, 9}:
            raise ValueError("--cell must contain 3 box lengths or 9 matrix values")
        array = np.asarray(parts, dtype=np.float64)
    else:
        array = np.asarray(value, dtype=np.float64)
    if array.shape == (3,):
        return np.diag(array)
    if array.size == 9:
        return array.reshape(3, 3)
    if array.shape == (3, 3):
        return array
    raise ValueError(f"Unsupported cell shape: {array.shape}")


def load_geometry_batches(
    geometries: str | Path,
    cell: np.ndarray | None = None,
    limit_geometries: int | None = None,
) -> list[GeometryBatch]:
    """Load one or more embedding results as frame-level geometry batches."""

    paths = _resolve_embedding_paths(geometries)
    batches: list[GeometryBatch] = []
    for path in paths:
        result = EmbeddingResult.load(path)
        frame_id = str(result.metadata.get("frame_id", result.metadata.get("frame_index", path.name)))
        positions = np.asarray(result.embedded_system, dtype=np.float64)
        grid_points = np.asarray(result.grid_points, dtype=np.float64)
        source_indices = np.arange(positions.shape[0], dtype=np.int64)
        if limit_geometries is not None:
            positions = positions[:limit_geometries]
            grid_points = grid_points[:limit_geometries]
            source_indices = source_indices[:limit_geometries]
        batch_cell = _cell_from_metadata(result.metadata)
        if batch_cell is None:
            batch_cell = cell
        batches.append(
            GeometryBatch(
                frame_id=frame_id,
                species=list(result.species),
                positions=positions,
                grid_points=grid_points,
                probe_indices=np.asarray(result.probe_indices, dtype=np.int64),
                cell=batch_cell,
                source_indices=source_indices,
                metadata={"embedding_dir": str(path), **dict(result.metadata)},
            )
        )
    return batches


def _resolve_embedding_paths(geometries: str | Path) -> list[Path]:
    path = Path(geometries)
    if path.is_file() and path.name == "embedded_geometries.npz":
        return [path.parent]
    if (path / "embedded_geometries.npz").exists():
        return [path]
    if path.is_dir():
        children = sorted(child for child in path.iterdir() if (child / "embedded_geometries.npz").exists())
        if children:
            return children
    raise FileNotFoundError(f"No embedded_geometries.npz found under {path}")


def _cell_from_metadata(metadata: dict[str, Any]) -> np.ndarray | None:
    for key in ("cell", "box", "cell_angstrom"):
        if key in metadata:
            return parse_cell(metadata[key])
    return None


def save_energy_result(result: EnergyResult, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "energies_hartree": result.energies_hartree,
        "grid_points": result.grid_points,
        "success_mask": result.success_mask,
    }
    if result.source_indices is not None:
        arrays["source_indices"] = result.source_indices
    np.savez_compressed(output_dir / "qc_energies.npz", **arrays)
    metadata = dict(result.metadata)
    metadata.update(
        {
            "backend": result.backend,
            "frame_id": result.frame_id,
            "geometry_count": int(result.energies_hartree.size),
            "success_count": int(result.success_mask.sum()),
            "files": {"arrays": "qc_energies.npz", "metadata": "metadata.json", "manifest": "manifest.jsonl"},
        }
    )
    if result.errors:
        metadata["errors"] = result.errors
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def append_manifest(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_xyz(path: str | Path, species: list[str], positions: np.ndarray, comment: str = "") -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(species)}\n")
        handle.write(f"{comment}\n")
        for element, xyz in zip(species, positions):
            handle.write(f"{element:2s} {xyz[0]:18.10f} {xyz[1]:18.10f} {xyz[2]:18.10f}\n")
