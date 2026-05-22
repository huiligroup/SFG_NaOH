#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
CONFIG="${ROOT}/config/visnet_initial_train.toml"
EXTRA_TRAIN_ARGS=()

collect_lines() {
  local target_name="$1"
  local line
  eval "$target_name=()"
  while IFS= read -r line; do
    eval "$target_name+=(\"\$line\")"
  done
}

usage() {
  cat <<'EOF'
Usage:
  bash NaOH_7.5M/scripts/train_visnet_initial.sh [--config PATH] [-- TRAIN_ARGS...]

Description:
  Run the initial ViSNet-eIP supervised training pipeline:
    1. preprocess CP2K trajectories into a pickle dataset
    2. precompute neighbor lists if enabled in the TOML
    3. launch train.py with --config

Options:
  --config PATH   Use a non-default TOML config
  --              Forward all remaining arguments to train.py
  -h, --help      Show this message

Environment:
  PYTHON          Python executable to use (default: python)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      [[ $# -ge 2 ]] || { echo "missing value for --config" >&2; exit 1; }
      CONFIG="$2"
      shift 2
      ;;
    --)
      shift
      EXTRA_TRAIN_ARGS=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

[[ -f "$CONFIG" ]] || { echo "config not found: $CONFIG" >&2; exit 1; }

"$PYTHON_BIN" - "$CONFIG" <<'PY'
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

config_path = Path(sys.argv[1])
with config_path.open("rb") as handle:
    data = tomllib.load(handle)

for section in ("preprocess", "edges", "train"):
    if section not in data:
        raise SystemExit(f"missing [{section}] table in {config_path}")

pre = data["preprocess"]
edges = data["edges"]
train = data["train"]

required_pre = ("source_root", "output", "trajectories")
required_train = ("data", "output_dir")
for key in required_pre:
    if key not in pre:
        raise SystemExit(f"missing preprocess.{key} in {config_path}")
for key in required_train:
    if key not in train:
        raise SystemExit(f"missing train.{key} in {config_path}")

if str(pre["output"]) != str(train["data"]):
    raise SystemExit(
        "config mismatch: preprocess.output must equal train.data so the shell pipeline "
        "does not preprocess one file and train on another"
    )

if edges.get("enabled", False):
    if "output" not in edges:
        raise SystemExit(f"missing edges.output in {config_path}")
    if "edge_data" not in train:
        raise SystemExit(
            "config mismatch: edges.enabled=true requires train.edge_data to point at the same file"
        )
    if str(edges["output"]) != str(train["edge_data"]):
        raise SystemExit(
            "config mismatch: edges.output must equal train.edge_data when edges.enabled=true"
        )

source_root = Path(pre["source_root"])
if not source_root.exists():
    raise SystemExit(f"preprocess.source_root does not exist: {source_root}")
PY

collect_lines OUTPUT_DIRS < <("$PYTHON_BIN" - "$CONFIG" <<'PY'
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

with Path(sys.argv[1]).open("rb") as handle:
    data = tomllib.load(handle)

pre = data["preprocess"]
edges = data["edges"]
train = data["train"]

print(Path(pre["output"]).parent)
if edges.get("enabled", False):
    print(Path(edges["output"]).parent)
print(Path(train["output_dir"]))
PY
)

for dir_path in "${OUTPUT_DIRS[@]}"; do
  mkdir -p "$dir_path"
done

collect_lines PREPROCESS_ARGS < <("$PYTHON_BIN" - "$CONFIG" <<'PY'
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

with Path(sys.argv[1]).open("rb") as handle:
    data = tomllib.load(handle)

pre = data["preprocess"]
args = [
    "--source-root", str(pre["source_root"]),
    "--output", str(pre["output"]),
    "--trajectories", *list(pre["trajectories"]),
]
if not pre.get("energy_check", True):
    args.append("--no-energy-check")
for arg in args:
    print(arg)
PY
)

echo "[1/3] preprocess -> ${PREPROCESS_ARGS[3]}"
"$PYTHON_BIN" "${ROOT}/visnet/data/preprocess.py" "${PREPROCESS_ARGS[@]}"

EDGE_ENABLED="$("$PYTHON_BIN" - "$CONFIG" <<'PY'
import sys
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]
with Path(sys.argv[1]).open("rb") as handle:
    data = tomllib.load(handle)
print("true" if data["edges"].get("enabled", False) else "false")
PY
)"

if [[ "$EDGE_ENABLED" == "true" ]]; then
  collect_lines EDGE_ARGS < <("$PYTHON_BIN" - "$CONFIG" <<'PY'
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

with Path(sys.argv[1]).open("rb") as handle:
    data = tomllib.load(handle)

pre = data["preprocess"]
edges = data["edges"]
args = [
    "--data", str(pre["output"]),
    "--output", str(edges["output"]),
    "--cutoff", str(edges["cutoff"]),
    "--max-num-neighbors", str(edges["max_num_neighbors"]),
    "--frame-stride", str(edges["frame_stride"]),
    "--num-workers", str(edges.get("num_workers", 1)),
    "--splits", *list(edges.get("splits", ["train", "val", "test"])),
]
limit = edges.get("limit")
if limit is not None:
    args.extend(["--limit", str(limit)])
for arg in args:
    print(arg)
PY
  )

  echo "[2/3] precompute_edges -> ${EDGE_ARGS[3]}"
  "$PYTHON_BIN" "${ROOT}/visnet/data/precompute_edges.py" "${EDGE_ARGS[@]}"
else
  echo "[2/3] precompute_edges skipped (edges.enabled=false)"
fi

echo "[3/3] train -> ${CONFIG}"
if [[ ${#EXTRA_TRAIN_ARGS[@]} -gt 0 ]]; then
  "$PYTHON_BIN" "${ROOT}/visnet/train.py" --config "$CONFIG" "${EXTRA_TRAIN_ARGS[@]}"
else
  "$PYTHON_BIN" "${ROOT}/visnet/train.py" --config "$CONFIG"
fi
