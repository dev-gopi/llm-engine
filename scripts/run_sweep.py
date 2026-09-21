"""Plan or explicitly execute a sequential training hyperparameter sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

import yaml

from training.sweeps import build_sweep, sweep_manifest
from utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--axes", type=Path, required=True, help="YAML mapping of axis to candidate values")
    parser.add_argument("--prefix", default="trial")
    parser.add_argument("--execute", action="store_true", help="run trials sequentially; default only prints plan")
    args = parser.parse_args()

    axes = load_yaml(args.axes)
    trials = build_sweep(load_yaml(args.training_config), axes, prefix=args.prefix)
    print(json.dumps(sweep_manifest(trials), sort_keys=True, indent=2))
    if not args.execute:
        return
    for trial in trials:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", encoding="utf-8", delete=False) as handle:
            yaml.safe_dump(trial.config, handle, sort_keys=True)
            trial_path = Path(handle.name)
        try:
            subprocess.run(
                [sys.executable, "scripts/train.py", "--training-config", str(trial_path)], check=True,
            )
        finally:
            trial_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
