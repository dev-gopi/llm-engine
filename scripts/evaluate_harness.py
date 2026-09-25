"""Run standardized LM Evaluation Harness benchmarks on model checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

import torch

from evaluation.harness import (
    EvaluationHarness,
    HarnessModelAdapter,
    HAS_LM_EVAL,
    list_tasks,
)
from model.gpt import MiniGPT
from model.vocabulary import adapt_config_to_tokenizer, checkpoint_tokenizer_options
from tokenizer.encoder import Tokenizer
from training.checkpoint import load_checkpoint
from utils.config import load_yaml
from utils.device import resolve_device
from utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Path to checkpoint .pt file")
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.gpu.yaml"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer"))
    parser.add_argument(
        "--tasks",
        default="mmlu,arc_challenge,gsm8k",
        help="Comma-separated list of benchmark tasks to run, or 'all'",
    )
    parser.add_argument("--num-fewshot", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--limit", type=int, default=None, help="Maximum evaluation samples per task")
    parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size")
    parser.add_argument("--device", default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema", help="Checkpoint weight flavor")
    parser.add_argument("--threads", type=int, default=4, help="CPU torch thread count")
    parser.add_argument("--output", type=Path, help="Atomically write the JSON evaluation result")
    parser.add_argument("--native-lm-eval", action="store_true", help="Run official lm_eval runner if installed")
    parser.add_argument("--list-tasks", action="store_true", help="List available tasks and exit")

    args = parser.parse_args()

    if args.list_tasks:
        available = list_tasks()
        print("Available harness benchmark tasks:")
        for t in available:
            print(f"  - {t}")
        return

    if not args.checkpoint:
        parser.error("--checkpoint is required unless --list-tasks is given")

    if args.threads < 1 or args.batch_size < 1:
        parser.error("threads and batch-size must be positive integers")

    if args.num_fewshot < 0:
        parser.error("num-fewshot must be non-negative")

    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive if specified")

    if args.output and args.output.exists():
        parser.error("output already exists; choose a new path to preserve evaluation evidence")

    torch.set_num_threads(args.threads)

    for label, path in (
        ("model configuration", args.model_config),
        ("tokenizer", args.tokenizer),
        ("checkpoint", args.checkpoint),
    ):
        exists = path.is_dir() if label == "tokenizer" else path.is_file()
        if not exists:
            parser.error(f"{label} not found: {path}")

    device = resolve_device(args.device)
    tokenizer = Tokenizer.load(args.tokenizer)
    model_config = load_yaml(args.model_config)
    try:
        model_config = adapt_config_to_tokenizer(model_config, tokenizer)
    except ValueError as error:
        parser.error(str(error))

    model = MiniGPT.from_config(model_config, device="cpu")
    checkpoint_info = load_checkpoint(
        args.checkpoint,
        model,
        use_ema=args.weights == "ema",
        restore_rng=False,
        **checkpoint_tokenizer_options(tokenizer, allow_extension=False),
    )

    model_adapter = HarnessModelAdapter(
        model,
        tokenizer,
        device=device,
        batch_size=args.batch_size,
    )

    all_registered = list_tasks()
    if args.tasks.strip().lower() == "all":
        selected_tasks = all_registered
    else:
        selected_tasks = [t.strip().lower() for t in args.tasks.split(",") if t.strip()]

    for t in selected_tasks:
        if t not in all_registered:
            parser.error(f"Task '{t}' is not recognized. Available: {all_registered}")

    if args.native_lm_eval:
        if not HAS_LM_EVAL:
            parser.error("lm_eval package is not installed; run without --native-lm-eval to use the built-in runner")
        import lm_eval
        logger.info("Executing official lm_eval.evaluator on tasks: %s", selected_tasks)
        eval_results = lm_eval.evaluator.simple_evaluate(
            model=model_adapter,
            tasks=selected_tasks,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
            batch_size=args.batch_size,
        )
        report_data = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(args.checkpoint),
            "step": checkpoint_info.get("step", 0),
            "weights_flavor": args.weights,
            "device": str(device),
            "tasks": selected_tasks,
            "num_fewshot": args.num_fewshot,
            "native_lm_eval": True,
            "results": eval_results,
        }
    else:
        harness = EvaluationHarness()
        report = harness.run(
            model_adapter,
            selected_tasks,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
            checkpoint_name=str(args.checkpoint),
            metadata={
                "step": checkpoint_info.get("step", 0),
                "weights_flavor": args.weights,
                "model_config": model_config,
                "device": str(device),
            },
        )
        report_data = report.to_dict()

    rendered = json.dumps(report_data, indent=2, ensure_ascii=False) + "\n"

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{args.output.name}.", dir=args.output.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(rendered)
            os.replace(temporary, args.output)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    print(rendered, end="")


if __name__ == "__main__":
    main()
