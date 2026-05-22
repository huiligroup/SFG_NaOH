#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
CONFIG="${ROOT}/config/visnet_active_learning.toml"

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
  bash NaOH_7.5M/scripts/run_visnet_active_learning.sh [--config PATH]

Description:
  Run online ViSNet-eIP active learning from an existing initial checkpoint.
  This script only validates required inputs and then launches run_active_learning.py.

Options:
  --config PATH   Use a non-default TOML config
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

collect_lines REQUIRED_PATHS < <("$PYTHON_BIN" - "$CONFIG" <<'PY'
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

with Path(sys.argv[1]).open("rb") as handle:
    data = tomllib.load(handle)

for section in ("model", "md", "cp2k", "retrain", "output"):
    if section not in data:
        raise SystemExit(f"missing [{section}] table in {sys.argv[1]}")

print(data["model"]["checkpoint"])
print(data["retrain"]["base_data"])
print(data["md"]["initial_xyz"])
print(data["cp2k"]["template_source"])
print(data["output"].get("root_dir", "NaOH_7.5M/data/active_learning"))
PY
)

CHECKPOINT="${REQUIRED_PATHS[0]}"
BASE_DATA="${REQUIRED_PATHS[1]}"
INITIAL_XYZ="${REQUIRED_PATHS[2]}"
CP2K_TEMPLATE="${REQUIRED_PATHS[3]}"
OUTPUT_ROOT="${REQUIRED_PATHS[4]}"

[[ -f "$CHECKPOINT" ]] || { echo "initial checkpoint not found: $CHECKPOINT" >&2; exit 1; }
[[ -f "$BASE_DATA" ]] || { echo "base supervised dataset not found: $BASE_DATA" >&2; exit 1; }
[[ -f "$INITIAL_XYZ" ]] || { echo "md.initial_xyz not found: $INITIAL_XYZ" >&2; exit 1; }
[[ -f "$CP2K_TEMPLATE" ]] || { echo "cp2k.template_source not found: $CP2K_TEMPLATE" >&2; exit 1; }

mkdir -p "$OUTPUT_ROOT"

echo "launch_active_learning config=$CONFIG checkpoint=$CHECKPOINT"
"$PYTHON_BIN" "${ROOT}/visnet/run_active_learning.py" --config "$CONFIG"
