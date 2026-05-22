"""Preprocess CP2K NaOH12 trajectories into a pickle dataset.

The output is intentionally plain NumPy data so it is easy to inspect and can
be consumed by either PyTorch Geometric training code or independent checks.
"""

from __future__ import annotations

import argparse
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Iterator

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAJECTORIES = ("NaOH12",)
DEFAULT_SOURCE_ROOT = ROOT / "data" / "NVE"
DEFAULT_OUTPUT = ROOT / "data" / "visnet" / "naoh12.pkl"
HARTREE_PER_BOHR_TO_HARTREE_PER_ANGSTROM = 1.8897261246257702
ELEMENT_Z = {"H": 1, "O": 8, "Na": 11}
ENERGY_RE = re.compile(r"\bi\s*=\s*(\d+).*?\bE\s*=\s*([-+0-9.Ee]+)")


def read_cell(input_file: Path) -> np.ndarray:
    vectors: dict[str, list[float]] = {}
    in_cell = False
    with input_file.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            upper = line.upper()
            if upper == "&CELL":
                in_cell = True
                continue
            if in_cell and upper.startswith("&END CELL"):
                break
            if in_cell:
                parts = line.split()
                if len(parts) == 4 and parts[0].upper() in {"A", "B", "C"}:
                    vectors[parts[0].upper()] = [float(v) for v in parts[1:]]
    if set(vectors) != {"A", "B", "C"}:
        raise ValueError(f"Could not parse A/B/C cell vectors from {input_file}")
    return np.asarray([vectors["A"], vectors["B"], vectors["C"]], dtype=np.float32)


def read_energies(ener_file: Path) -> dict[int, float]:
    energies: dict[int, float] = {}
    with ener_file.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            energies[int(parts[0])] = float(parts[4])
    return energies


def iter_xyz_frames(path: Path) -> Iterator[tuple[int, float, list[str], np.ndarray]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        while True:
            count_line = handle.readline()
            if not count_line:
                return
            count_line = count_line.strip()
            if not count_line:
                continue
            natoms = int(count_line)
            comment = handle.readline()
            match = ENERGY_RE.search(comment)
            if match is None:
                raise ValueError(f"Could not parse frame index/energy in {path}: {comment!r}")
            frame_idx = int(match.group(1))
            energy = float(match.group(2))
            species: list[str] = []
            values = np.empty((natoms, 3), dtype=np.float32)
            for atom_idx in range(natoms):
                parts = handle.readline().split()
                if len(parts) < 4:
                    raise ValueError(f"Malformed atom line {atom_idx} in {path}")
                species.append(parts[0])
                values[atom_idx] = (float(parts[1]), float(parts[2]), float(parts[3]))
            yield frame_idx, energy, species, values


def split_labels(nframes: int) -> np.ndarray:
    labels = np.empty(nframes, dtype=object)
    train_end = int(0.8 * nframes)
    val_end = int(0.9 * nframes)
    labels[:train_end] = "train"
    labels[train_end:val_end] = "val"
    labels[val_end:] = "test"
    return labels


def _single_match(directory: Path, patterns: tuple[str, ...], label: str) -> Path:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(directory.glob(pattern)))
    unique = list(dict.fromkeys(matches))
    if not unique:
        raise FileNotFoundError(f"No {label} file found in {directory}; tried patterns={patterns}")
    if len(unique) > 1:
        raise ValueError(f"Multiple {label} files found in {directory}: {unique}")
    return unique[0]


def process_trajectory(traj_dir: Path, name: str, check_energy: bool = True) -> dict[str, object]:
    pos_file = _single_match(traj_dir, ("*-pos-1.xyz",), "position XYZ")
    frc_file = _single_match(traj_dir, ("*-frc-1.xyz",), "force XYZ")
    ener_file = _single_match(traj_dir, ("*.ener",), "energy")
    input_file = _single_match(traj_dir, ("*.inp",), "CP2K input")
    for path in (pos_file, frc_file, ener_file, input_file):
        if not path.exists():
            raise FileNotFoundError(path)

    cell = read_cell(input_file)
    energy_by_step = read_energies(ener_file)
    pos_iter = iter_xyz_frames(pos_file)
    frc_iter = iter_xyz_frames(frc_file)

    species_ref: list[str] | None = None
    positions: list[np.ndarray] = []
    forces: list[np.ndarray] = []
    energies: list[float] = []
    frames: list[int] = []
    max_energy_delta = 0.0

    for pos_frame in pos_iter:
        try:
            frc_frame = next(frc_iter)
        except StopIteration as exc:
            raise ValueError(f"Force trajectory ended before position trajectory for {name}") from exc

        p_step, p_energy, p_species, pos = pos_frame
        f_step, f_energy, f_species, force = frc_frame
        if p_step != f_step:
            raise ValueError(f"Frame mismatch in {name}: positions step {p_step}, forces step {f_step}")
        if p_species != f_species:
            raise ValueError(f"Species mismatch at trajectory {name}, step {p_step}")
        if species_ref is None:
            species_ref = p_species
        elif species_ref != p_species:
            raise ValueError(f"Species ordering changed in trajectory {name}, step {p_step}")

        if check_energy:
            if p_step not in energy_by_step:
                raise ValueError(f"Step {p_step} from XYZ is missing in {ener_file}")
            max_energy_delta = max(max_energy_delta, abs(energy_by_step[p_step] - p_energy), abs(f_energy - p_energy))

        positions.append(pos)
        forces.append(force * HARTREE_PER_BOHR_TO_HARTREE_PER_ANGSTROM)
        energies.append(p_energy)
        frames.append(p_step)

    try:
        extra_force_frame = next(frc_iter)
    except StopIteration:
        extra_force_frame = None
    if extra_force_frame is not None:
        raise ValueError(f"Force trajectory has extra frames for {name}")
    if species_ref is None:
        raise ValueError(f"No frames found in {pos_file}")

    species_counts = Counter(species_ref)
    expected = Counter({"H": 308, "O": 160, "Na": 12})
    if species_counts != expected:
        raise ValueError(f"Unexpected species counts for {name}: {species_counts}, expected {expected}")

    return {
        "traj": name,
        "species": np.asarray(species_ref, dtype=object),
        "z": np.asarray([ELEMENT_Z[s] for s in species_ref], dtype=np.int64),
        "pos": np.stack(positions).astype(np.float32, copy=False),
        "forces": np.stack(forces).astype(np.float32, copy=False),
        "energy": np.asarray(energies, dtype=np.float64),
        "cell": cell,
        "frame": np.asarray(frames, dtype=np.int64),
        "split": split_labels(len(frames)),
        "max_energy_delta": max_energy_delta,
    }


def build_dataset(source_root: Path, trajectories: tuple[str, ...], check_energy: bool) -> dict[str, object]:
    processed = [process_trajectory(source_root / traj, traj, check_energy=check_energy) for traj in trajectories]
    z0 = processed[0]["z"]
    species0 = processed[0]["species"]
    for item in processed[1:]:
        if not np.array_equal(z0, item["z"]):
            raise ValueError("Atomic number ordering differs across trajectories")
        if not np.array_equal(species0, item["species"]):
            raise ValueError("Species ordering differs across trajectories")

    return {
        "z": z0,
        "species": species0,
        "pos": np.concatenate([item["pos"] for item in processed], axis=0),
        "forces": np.concatenate([item["forces"] for item in processed], axis=0),
        "energy": np.concatenate([item["energy"] for item in processed], axis=0),
        "cell": np.concatenate(
            [np.repeat(item["cell"][None, :, :], len(item["frame"]), axis=0) for item in processed],
            axis=0,
        ),
        "traj": np.concatenate([np.full(len(item["frame"]), item["traj"], dtype=object) for item in processed]),
        "frame": np.concatenate([item["frame"] for item in processed]),
        "split": np.concatenate([item["split"] for item in processed]),
        "units": {
            "pos": "angstrom",
            "cell": "angstrom",
            "energy": "hartree",
            "forces": "hartree/angstrom",
            "source_forces": "hartree/bohr",
        },
        "metadata": {
            "trajectories": trajectories,
            "natoms": int(len(z0)),
            "species_counts": dict(Counter(species0)),
            "force_conversion": HARTREE_PER_BOHR_TO_HARTREE_PER_ANGSTROM,
            "max_energy_delta": max(float(item["max_energy_delta"]) for item in processed),
            "frames_by_trajectory": {item["traj"]: int(len(item["frame"])) for item in processed},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trajectories", nargs="+", default=list(DEFAULT_TRAJECTORIES))
    parser.add_argument("--no-energy-check", action="store_true")
    args = parser.parse_args()

    dataset = build_dataset(
        source_root=args.source_root,
        trajectories=tuple(args.trajectories),
        check_energy=not args.no_energy_check,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    meta = dataset["metadata"]
    print(f"Wrote {args.output}")
    print(f"frames={len(dataset['energy'])} natoms={meta['natoms']} counts={meta['species_counts']}")
    print(f"frames_by_trajectory={meta['frames_by_trajectory']}")
    print(f"max_energy_delta={meta['max_energy_delta']:.3e} hartree")


if __name__ == "__main__":
    main()
