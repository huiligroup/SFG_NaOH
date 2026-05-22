"""Extract molecule descriptors from a trained ViSNet-eIP checkpoint."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from visnet.model import VisNetEIP  # noqa: E402


DEFAULT_DATA = ROOT / "data" / "visnet" / "naoh12.pkl"
DEFAULT_CHECKPOINT = ROOT / "visnet" / "runs" / "naoh12_visnet" / "best.pt"
DEFAULT_OUTPUT = ROOT / "src" / "descriptor" / "mol_features.pkl"


def parse_mol_ids(raw: str, max_mol_id: int) -> list[int]:
    if raw.lower() == "all":
        return list(range(max_mol_id + 1))
    mol_ids: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", maxsplit=1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid molecule id range: {item}")
            mol_ids.extend(range(start, end + 1))
        else:
            mol_ids.append(int(item))
    mol_ids = list(dict.fromkeys(mol_ids))
    bad = [mol_id for mol_id in mol_ids if mol_id < 0 or mol_id > max_mol_id]
    if bad:
        raise ValueError(f"Molecule ids out of range 0-{max_mol_id}: {bad}")
    return mol_ids


def build_default_molecules(species: np.ndarray) -> list[dict[str, object]]:
    """Return fixed molecule definitions for the NaOH12 trajectory atom order."""

    if len(species) != 480:
        raise ValueError(f"Expected 480 atoms for NaOH12, got {len(species)}")
    molecules: list[dict[str, object]] = []
    for mol_id in range(148):
        atom_indices = [3 * mol_id, 3 * mol_id + 1, 3 * mol_id + 2]
        mol_species = [str(species[idx]) for idx in atom_indices]
        if mol_species != ["O", "H", "H"]:
            raise ValueError(f"Water molecule {mol_id} does not match O,H,H: {mol_species}")
        molecules.append(
            {
                "mol_id": mol_id,
                "kind": "water",
                "atom_indices": atom_indices,
                "species": mol_species,
            }
        )
    offset = 148 * 3
    for local_id in range(12):
        mol_id = 148 + local_id
        atom_indices = [offset + 3 * local_id, offset + 3 * local_id + 1, offset + 3 * local_id + 2]
        mol_species = [str(species[idx]) for idx in atom_indices]
        if mol_species != ["Na", "O", "H"]:
            raise ValueError(f"NaOH molecule {mol_id} does not match Na,O,H: {mol_species}")
        molecules.append(
            {
                "mol_id": mol_id,
                "kind": "naoh",
                "atom_indices": atom_indices,
                "species": mol_species,
            }
        )
    return molecules


def infer_frame_step_by_traj(traj: np.ndarray, frame: np.ndarray) -> dict[str, int | None]:
    steps: dict[str, int | None] = {}
    for name in np.unique(traj):
        values = frame[traj == name]
        if len(values) < 2:
            steps[str(name)] = None
            continue
        diffs = np.diff(values.astype(np.int64))
        positive = diffs[diffs > 0]
        steps[str(name)] = int(np.median(positive)) if positive.size else None
    return steps


def selected_frame_indices(dataset: dict, split: str, frame_stride: int, limit_frames: int | None) -> np.ndarray:
    if frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    if split == "all":
        indices = np.arange(len(dataset["frame"]), dtype=np.int64)
    else:
        indices = np.flatnonzero(np.asarray(dataset["split"]) == split)
    indices = indices[::frame_stride]
    if limit_frames is not None:
        indices = indices[:limit_frames]
    if len(indices) == 0:
        raise ValueError(f"No frames selected for split={split!r}")
    return indices


def load_checkpoint(path: Path, device: torch.device) -> tuple[VisNetEIP, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    args = checkpoint.get("args", {})
    model = VisNetEIP(
        hidden_channels=int(args.get("hidden_channels", 64)),
        num_layers=int(args.get("num_layers", 6)),
        num_heads=int(args.get("num_heads", 8)),
        num_rbf=int(args.get("num_rbf", 32)),
        cutoff=float(args.get("cutoff", 6.0)),
        max_num_neighbors=int(args.get("max_num_neighbors", 64)),
        mean=float(checkpoint.get("energy_mean", 0.0)),
        std=float(checkpoint.get("energy_std", 1.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, {
        "hidden_channels": int(args.get("hidden_channels", 64)),
        "num_layers": int(args.get("num_layers", 6)),
        "num_heads": int(args.get("num_heads", 8)),
        "num_rbf": int(args.get("num_rbf", 32)),
        "cutoff": float(args.get("cutoff", 6.0)),
        "max_num_neighbors": int(args.get("max_num_neighbors", 64)),
        "energy_mean": float(checkpoint.get("energy_mean", 0.0)),
        "energy_std": float(checkpoint.get("energy_std", 1.0)),
        "epoch": checkpoint.get("epoch"),
        "train_step": checkpoint.get("train_step"),
        "best_metric": checkpoint.get("best_metric"),
    }


def chunks(values: np.ndarray, size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--mol-ids", default="148", help="Comma/range list such as 148,149 or 148-159, or all.")
    parser.add_argument("--split", default="all", choices=("all", "train", "val", "test"))
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    device = torch.device(args.device)
    with args.data.open("rb") as handle:
        dataset = pickle.load(handle)
    species = np.asarray(dataset["species"], dtype=object)
    molecules = build_default_molecules(species)
    mol_ids = parse_mol_ids(args.mol_ids, max_mol_id=len(molecules) - 1)
    mol_lookup = {int(item["mol_id"]): item for item in molecules}
    frame_indices = selected_frame_indices(dataset, args.split, args.frame_stride, args.limit_frames)

    model, model_info = load_checkpoint(args.checkpoint, device)
    z_single = torch.as_tensor(dataset["z"], dtype=torch.long, device=device)
    natoms = int(z_single.numel())

    features: list[np.ndarray] = []
    mol_id_rows: list[int] = []
    atom_index_rows: list[list[int]] = []
    traj_rows: list[str] = []
    frame_rows: list[int] = []
    frame_index_rows: list[int] = []
    pos_rows: list[np.ndarray] = []
    cell_rows: list[np.ndarray] = []
    record_frame_pos_index: list[int] = []
    system_pos: list[np.ndarray] = []
    system_cell: list[np.ndarray] = []
    unique_frame_indices: list[int] = []

    with torch.no_grad():
        for batch_number, batch_indices in enumerate(chunks(frame_indices, args.batch_size), start=1):
            batch_len = len(batch_indices)
            z = z_single.repeat(batch_len)
            pos = torch.as_tensor(dataset["pos"][batch_indices], dtype=torch.float32, device=device).reshape(
                batch_len * natoms, 3
            )
            cell = torch.as_tensor(dataset["cell"][batch_indices], dtype=torch.float32, device=device)
            batch = torch.repeat_interleave(torch.arange(batch_len, device=device), natoms)
            atom_features = model.extract_atom_features(z=z, pos=pos, batch=batch, cell=cell)["atom_features"]
            atom_features = atom_features.detach().cpu().numpy().astype(np.float32, copy=False)

            for local_frame, frame_idx in enumerate(batch_indices):
                frame_feature_start = local_frame * natoms
                frame_feature_end = frame_feature_start + natoms
                frame_features = atom_features[frame_feature_start:frame_feature_end]
                frame_pos_index = len(system_pos)
                system_pos.append(np.asarray(dataset["pos"][frame_idx], dtype=np.float32))
                system_cell.append(np.asarray(dataset["cell"][frame_idx], dtype=np.float32))
                unique_frame_indices.append(int(frame_idx))
                for mol_id in mol_ids:
                    mol = mol_lookup[mol_id]
                    atom_indices = list(mol["atom_indices"])
                    features.append(frame_features[atom_indices].reshape(-1))
                    mol_id_rows.append(mol_id)
                    atom_index_rows.append(atom_indices)
                    traj_rows.append(str(dataset["traj"][frame_idx]))
                    frame_rows.append(int(dataset["frame"][frame_idx]))
                    frame_index_rows.append(int(frame_idx))
                    pos_rows.append(np.asarray(dataset["pos"][frame_idx][atom_indices], dtype=np.float32))
                    cell_rows.append(np.asarray(dataset["cell"][frame_idx], dtype=np.float32))
                    record_frame_pos_index.append(frame_pos_index)
            if batch_number % 10 == 0:
                print(f"extracted_batches={batch_number} records={len(features)}")

    output = {
        "features": np.stack(features).astype(np.float32, copy=False),
        "mol_ids": np.asarray(mol_id_rows, dtype=np.int64),
        "atom_indices": np.asarray(atom_index_rows, dtype=np.int64),
        "species": species,
        "mol_species": np.asarray([[str(species[idx]) for idx in row] for row in atom_index_rows], dtype=object),
        "traj": np.asarray(traj_rows, dtype=object),
        "frame": np.asarray(frame_rows, dtype=np.int64),
        "frame_index": np.asarray(frame_index_rows, dtype=np.int64),
        "pos": np.stack(pos_rows).astype(np.float32, copy=False),
        "cell": np.stack(cell_rows).astype(np.float32, copy=False),
        "system_pos": np.stack(system_pos).astype(np.float32, copy=False),
        "system_cell": np.stack(system_cell).astype(np.float32, copy=False),
        "system_frame_index": np.asarray(unique_frame_indices, dtype=np.int64),
        "record_frame_pos_index": np.asarray(record_frame_pos_index, dtype=np.int64),
        "checkpoint": str(args.checkpoint),
        "feature_type": "visnet_scalar_x_before_energy_head",
        "source_data": str(args.data),
        "source_frame_step_by_traj": infer_frame_step_by_traj(np.asarray(dataset["traj"]), np.asarray(dataset["frame"])),
        "selected_split": args.split,
        "frame_stride_on_pkl": int(args.frame_stride),
        "model_args": model_info,
        "molecule_definitions": molecules,
        "units": dataset.get("units", {}),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"wrote_features={args.output} records={output['features'].shape[0]} "
        f"feature_dim={output['features'].shape[1]} frames={len(frame_indices)} mol_ids={mol_ids}"
    )


if __name__ == "__main__":
    main()
