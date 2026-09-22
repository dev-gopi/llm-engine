"""Estimate model/attention-state memory across deployment precisions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

from runtime.resource_planner import deployment_matrix
from utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.gpu.yaml"))
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--memory-gib", type=float, help="optional device/RAM budget in GiB")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    budget = None if args.memory_gib is None else int(args.memory_gib * 1024 ** 3)
    payload = {
        "schema_version": 1,
        "model_config": str(args.model_config),
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "memory_budget_bytes": budget,
        "plans": deployment_matrix(
            load_yaml(args.model_config),
            context_length=args.context_length,
            batch_size=args.batch_size,
            memory_budget_bytes=budget,
        ),
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
