"""Run ViSNet-eIP online active learning with ASE + UDD bias and CP2K labeling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from visnet.active_learning import ActiveLearningConfig, run_active_learning
else:
    from .active_learning import ActiveLearningConfig, run_active_learning


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="TOML config under NaOH_7.5M/config/")
    args = parser.parse_args()
    config = ActiveLearningConfig.from_toml(args.config)
    summary = run_active_learning(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
