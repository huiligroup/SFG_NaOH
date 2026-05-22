"""Append newly labeled active-learning frames to a ViSNet training pickle."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .types import LabeledFrame


def append_labeled_frames(
    base_data_path: str | Path,
    labeled_frames: list[LabeledFrame],
    output_path: str | Path,
    *,
    traj_name: str,
) -> Path:
    base_data_path = Path(base_data_path)
    output_path = Path(output_path)
    with base_data_path.open("rb") as handle:
        dataset = pickle.load(handle)

    if not labeled_frames:
        raise ValueError("No labeled frames to append")

    base_species = [str(item) for item in np.asarray(dataset["species"], dtype=object).tolist()]
    base_z = np.asarray(dataset["z"], dtype=np.int64)
    base_next_frame = int(np.asarray(dataset["frame"], dtype=np.int64).max()) + 1

    for index, frame in enumerate(labeled_frames):
        if list(frame.species) != base_species:
            raise ValueError(f"Labeled frame {index} species ordering does not match base dataset")
        if frame.positions.shape != np.asarray(dataset["pos"][0]).shape:
            raise ValueError(f"Labeled frame {index} positions shape does not match base dataset")

    new_pos = np.stack([frame.positions for frame in labeled_frames]).astype(np.float32, copy=False)
    new_forces = np.stack([frame.forces_hartree_per_angstrom for frame in labeled_frames]).astype(np.float32, copy=False)
    new_energy = np.asarray([frame.energy_hartree for frame in labeled_frames], dtype=np.float64)
    new_cell = np.stack([frame.cell for frame in labeled_frames]).astype(np.float32, copy=False)
    new_frame_numbers = np.asarray([base_next_frame + index for index in range(len(labeled_frames))], dtype=np.int64)
    new_traj = np.asarray([traj_name] * len(labeled_frames), dtype=object)
    new_split = np.asarray(["train"] * len(labeled_frames), dtype=object)

    updated = {
        "z": base_z,
        "species": np.asarray(base_species, dtype=object),
        "pos": np.concatenate([np.asarray(dataset["pos"]), new_pos], axis=0),
        "forces": np.concatenate([np.asarray(dataset["forces"]), new_forces], axis=0),
        "energy": np.concatenate([np.asarray(dataset["energy"]), new_energy], axis=0),
        "cell": np.concatenate([np.asarray(dataset["cell"]), new_cell], axis=0),
        "traj": np.concatenate([np.asarray(dataset["traj"], dtype=object), new_traj], axis=0),
        "frame": np.concatenate([np.asarray(dataset["frame"], dtype=np.int64), new_frame_numbers], axis=0),
        "split": np.concatenate([np.asarray(dataset["split"], dtype=object), new_split], axis=0),
        "units": dict(dataset.get("units", {})),
        "metadata": dict(dataset.get("metadata", {})),
    }
    updated["metadata"]["active_learning_appended_frames"] = updated["metadata"].get("active_learning_appended_frames", 0) + len(labeled_frames)
    updated["metadata"]["last_active_learning_traj"] = traj_name

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(updated, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return output_path
