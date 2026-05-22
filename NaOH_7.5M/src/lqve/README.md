# LQVE workflow skeleton

This directory is the new organized workspace for the Local Quantum Vibration
Embedding (LQVE) method. The historical scripts and generated files remain in
`../lqve_old` as reference material only. New code, inputs, intermediate data,
and analysis products should be created under this directory.

## Method layout

The workflow has two main stages.

1. Reference DVR stage

   This stage starts from a high-accuracy reference potential energy surface
   (PES). It solves the reference vibrational problem and produces reference
   vibrational levels, wavefunctions, and DVR/PODVR points.

2. LQVE perturbation stage

   This stage embeds the reference DVR points into real-time molecular
   environments. It evaluates the perturbed potential on those points and uses
   perturbation theory or an effective Hamiltonian to estimate instantaneous
   vibrational frequencies.

## Directory map

```text
lqve/
  README.md
  configs/                    Runtime configuration files.
  scripts/                    Ordered command-line entry points.
  lqve/                       Reusable Python modules.
    dvr/                      SINC-DVR, PODVR, PES interpolation, reference states.
    embedding/                Alignment, rotation/translation removal, DVR geometry insertion.
    perturbation/             RSPT, effective Hamiltonian, frequency conversion.
    qc/                       Quantum chemistry input generation and output parsing.
    analysis/                 Frequency statistics, diagnostics, and plots.
    io/                       XYZ, CSV, NPY/PKL, and configuration I/O.
    utils/                    Units, atomic masses, logging, and path helpers.
  data/
    raw/                      Raw local inputs copied or linked for a run.
    processed/                Cleaned and normalized inputs.
    dvr/                      Generated reference DVR arrays.
    embedded_geometries/      DVR-point geometries embedded in MD environments.
    qc_outputs/               Parsed or raw quantum chemistry energy outputs.
    shifts/                   Instantaneous frequencies and frequency shifts.
  references/
    structures/               Reference chromophore and cluster structures.
    pes/                      Reference PES tables or fitted surfaces.
    modes/                    Normal modes and mass-weighted mode vectors.
    dvr_points/               Curated reference DVR/PODVR point files.
  results/
    tables/                   Final CSV/PKL/NPY summaries.
    figures/                  Figures for diagnostics and reports.
    logs/                     Runtime logs.
```

## Ordered entry points

Run scripts in numeric order as implementations are filled in:

```bash
python NaOH_7.5M/src/lqve/scripts/01_build_reference_dvr.py --help
python NaOH_7.5M/src/lqve/scripts/02_build_embedded_geometries.py --help
python NaOH_7.5M/src/lqve/scripts/03_run_qc_energies.py --help
python NaOH_7.5M/src/lqve/scripts/04_compute_lqve_shift.py --help
python NaOH_7.5M/src/lqve/scripts/05_analyze_results.py --help
```

All five ordered scripts are implemented. `05_analyze_results.py` reads the
structured shift outputs and generates final tables, figures, and a Markdown
report.

Example DVR command:

```bash
python NaOH_7.5M/src/lqve/scripts/01_build_reference_dvr.py \
  --pes NaOH_7.5M/src/lqve_old/get_DVR_points/energy_9115_minus_1_to_1_8000.dat \
  --dims 3 \
  --mode-names q1,q2,q3 \
  --ranges -0.7:0.7,-0.9:0.9,-0.6:0.6 \
  --sinc-points 200,200,200 \
  --podvr-points 7,10,7 \
  --states 80 \
  --coord-unit angstrom \
  --energy-unit cm-1 \
  --output-dir NaOH_7.5M/src/lqve/data/dvr/naoh_3d_7_10_7
```

The DVR stage writes structured output only:

- `dvr_result.npz`: levels in Hartree, wavefunctions, DVR grids in Bohr, basis
  shape, and state count.
- `metadata.json`: input PES, units, mode settings, solver, sparse matrix size,
  nonzero count, and runtime.

Example embedding command:

```bash
python NaOH_7.5M/src/lqve/scripts/02_build_embedded_geometries.py \
  --reference-xyz NaOH_7.5M/src/lqve_old/ref/h2o_ref.xyz \
  --system-xyz NaOH_7.5M/src/lqve_old/ref/one_cl_9115_ref.xyz \
  --probe-indices 0,1,2 \
  --modes NaOH_7.5M/src/lqve/references/modes/h2o_modes.npy \
  --dvr-data NaOH_7.5M/src/lqve/data/dvr/naoh_3d_7_10_7 \
  --grid-unit bohr \
  --mode-unit angstrom_per_bohr \
  --output-dir NaOH_7.5M/src/lqve/data/embedded_geometries/one_cl_9115 \
  --write-xyz
```

If multiple reference DVR/PES sets have already been prepared, the embedding
stage can select the closest reference for each frame from a ViSNet descriptor
library before constructing DVR geometries:

```bash
python NaOH_7.5M/src/lqve/scripts/02_build_embedded_geometries.py \
  --system-xyz NaOH_7.5M/src/lqve/data/processed/frame.xyz \
  --probe-indices 444,445,446 \
  --reference-library NaOH_7.5M/src/lqve/references/reference_library.json \
  --checkpoint NaOH_7.5M/visnet/runs/naoh12_visnet/best.pt \
  --feature-data NaOH_7.5M/data/visnet/naoh12.pkl \
  --frame-index 0 \
  --output-dir NaOH_7.5M/src/lqve/data/embedded_geometries/frame_000000
```

`reference_library.json` stores only precomputed references. Each entry must
map a reference descriptor to the LQVE assets needed downstream:
`reference_xyz`, `modes`, `dvr_data`, and `reference_energies`. The
runtime selection does not rebuild PES surfaces or solve DVR; it only chooses
which prepared reference best matches the current probe descriptor. The selected
entry is written to `selected_reference.json` and copied into `metadata.json`.

The embedding stage writes:

- `embedded_geometries.npz`: embedded probe/system coordinates, DVR grid
  combinations, probe indices, reference/system coordinates, rotation, and
  translation.
- `metadata.json`: input paths, units, dimensions, grid sizes, projection
  diagnostics, atom counts, and runtime.
- optional `xyz/` and `xyz_index.csv` when `--write-xyz` is set.

Example ViSNet energy command:

```bash
python NaOH_7.5M/src/lqve/scripts/03_run_qc_energies.py \
  --geometries NaOH_7.5M/src/lqve/data/embedded_geometries/one_cl_9115 \
  --backend visnet \
  --checkpoint NaOH_7.5M/visnet/runs/naoh12_visnet/best.pt \
  --cell 16.63,16.63,44.10 \
  --device cuda \
  --batch-size 16 \
  --output-dir NaOH_7.5M/src/lqve/data/qc_outputs/one_cl_9115_visnet \
  --resume
```

Example external xTB command:

```bash
python NaOH_7.5M/src/lqve/scripts/03_run_qc_energies.py \
  --geometries NaOH_7.5M/src/lqve/data/embedded_geometries/one_cl_9115 \
  --backend xtb \
  --command "xtb --gfn 2" \
  --workers 4 \
  --batch-size 1 \
  --output-dir NaOH_7.5M/src/lqve/data/qc_outputs/one_cl_9115_xtb \
  --keep-workdirs \
  --resume
```

The energy stage treats each embedding result as an independent frame. It saves
one frame directory under the requested output directory:

- `qc_energies.npz`: `energies_hartree`, `grid_points`, `success_mask`, and
  `source_indices`.
- `metadata.json`: backend, checkpoint or command, timing, frame id, success
  count, and input embedding path.
- root `manifest.jsonl`: frame/chunk status entries for resume and debugging.

The ViSNet backend is the preferred production path for this project. It reads
the full embedded box directly from `embedded_geometries.npz` and evaluates DVR
grid geometries in GPU batches. Gaussian, CP2K, and xTB backends are retained for
reference calculations and generate temporary input files only when needed.

Example shift command:

```bash
python NaOH_7.5M/src/lqve/scripts/04_compute_lqve_shift.py \
  --dvr-data NaOH_7.5M/src/lqve/data/dvr/naoh_3d_7_10_7 \
  --qc-outputs NaOH_7.5M/src/lqve/data/qc_outputs/one_cl_9115_visnet \
  --reference-energies NaOH_7.5M/src/lqve/references/pes/reference_grid_energies_hartree.npy \
  --output-dir NaOH_7.5M/src/lqve/data/shifts \
  --run-name one_cl_9115_visnet \
  --n-contract 30 \
  --n-transitions 6
```

When QC outputs come from dynamically selected references, the shift stage can
omit `--dvr-data` and `--reference-energies`; it will read the per-frame
`selected_reference` metadata propagated from embedding through QC:

```bash
python NaOH_7.5M/src/lqve/scripts/04_compute_lqve_shift.py \
  --qc-outputs NaOH_7.5M/src/lqve/data/qc_outputs/dynamic_visnet \
  --output-dir NaOH_7.5M/src/lqve/data/shifts \
  --run-name dynamic_visnet \
  --n-contract 30 \
  --n-transitions 6
```

The shift stage writes:

- one directory per frame containing `shift_result.npz` and `metadata.json`;
- `shifts_summary.csv` with frame id, success flag, selected `reference_id`,
  transitions in `cm^-1`, and shifts in `cm^-1`;
- `shifts_summary.npz` for array-based analysis;
- `manifest.jsonl` for frame-level status and source QC paths.

For smoke testing only, `--reference-mode first-qc` can use the first QC frame
as the reference energy grid. Production calculations should pass an explicit
`--reference-energies` file in Hartree.

Example analysis command:

```bash
python NaOH_7.5M/src/lqve/scripts/05_analyze_results.py \
  --shifts NaOH_7.5M/src/lqve/data/shifts/one_cl_9115_visnet \
  --output-dir NaOH_7.5M/src/lqve/results \
  --run-name one_cl_9115_visnet \
  --plot-format png
```

The analysis stage writes:

- `tables/transition_stats.csv`, `tables/shift_stats.csv`, and
  `tables/frame_quality.csv`;
- `figures/transitions_timeseries.png`, `figures/shifts_timeseries.png`,
  `figures/transition_histograms.png`, `figures/shift_histograms.png`,
  `figures/transition_correlation.png`, and `figures/frame_quality.png`;
- `report.md` with paths, success rate, summary statistics, figures, and failed
  or incomplete frame records.

## Relationship to `lqve_old`

`../lqve_old` contains the previous implementation, including historical DVR
scripts, perturbation scripts, quantum chemistry runners, generated energy
tables, plots, and other outputs. Do not add new products there. Future
refactoring should copy only the needed logic into `lqve/lqve/` modules and keep
data products inside `data/`, `references/`, or `results/`.
