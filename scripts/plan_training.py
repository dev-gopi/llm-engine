"""Estimate LLM training tokens, steps, FLOPs, memory, runtime, and cost."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

from training.planner import plan_training
from utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--training-tokens", type=int)
    parser.add_argument("--bytes-per-token", type=float, default=4.0)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--hardware-tflops", type=float)
    parser.add_argument("--utilization", type=float, default=0.35)
    parser.add_argument("--gpu-memory-gib", type=float)
    parser.add_argument("--hourly-cost-per-gpu", type=float)
    parser.add_argument("--require-fit", action="store_true")
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--max-cost", type=float)
    args = parser.parse_args()
    model_config = load_yaml(args.model_config)
    training_config = load_yaml(args.training_config)
    input_bytes = sum(
        Path(path).stat().st_size
        for path in training_config.get("train_files", [])
        if Path(path).is_file()
    )
    if args.training_tokens is None:
        if input_bytes == 0:
            parser.error("no readable train files; provide --training-tokens")
        if args.bytes_per_token <= 0:
            parser.error("bytes per token must be positive")
        tokens_per_epoch = math.ceil(input_bytes / args.bytes_per_token)
        training_tokens = tokens_per_epoch * int(training_config.get("epochs", 1))
        token_source = "estimated_from_input_bytes"
    else:
        training_tokens = args.training_tokens
        tokens_per_epoch = None
        token_source = "explicit"
    plan = plan_training(
        model_config, training_config, training_tokens=training_tokens, gpus=args.gpus,
        hardware_tflops=args.hardware_tflops, utilization=args.utilization,
        gpu_memory_gib=args.gpu_memory_gib, hourly_cost_per_gpu=args.hourly_cost_per_gpu,
    ).to_dict()
    plan["token_estimate"] = {
        "source": token_source, "input_bytes": input_bytes,
        "bytes_per_token": args.bytes_per_token if token_source != "explicit" else None,
        "tokens_per_epoch": tokens_per_epoch,
    }
    violations = []
    if args.require_fit and plan["fits_memory"] is not True:
        violations.append("estimated peak memory does not fit the supplied GPU memory")
    if args.max_hours is not None and (
        plan["estimated_hours"] is None or plan["estimated_hours"] > args.max_hours
    ):
        violations.append("estimated runtime exceeds --max-hours or cannot be calculated")
    if args.max_cost is not None and (
        plan["estimated_cost"] is None or plan["estimated_cost"] > args.max_cost
    ):
        violations.append("estimated cost exceeds --max-cost or cannot be calculated")
    plan["constraints"] = {"passed": not violations, "violations": violations}
    print(json.dumps(plan, indent=2))
    if violations:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
