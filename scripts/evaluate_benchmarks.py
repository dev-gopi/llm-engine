"""Run deterministic held-out generation checks grouped by capability."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

import torch

from evaluation.benchmarks import BenchmarkCase, score_answer, summarize_scores, compare_reports
from inference.context import format_system_prompt
from inference.generator import Generator
from model.gpt import MiniGPT
from model.vocabulary import adapt_config_to_tokenizer, checkpoint_tokenizer_options
from tokenizer.encoder import Tokenizer
from training.checkpoint import load_checkpoint
from utils.config import load_yaml
from utils.device import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("configs/evaluation.core.jsonl"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.gpu.yaml"))
    parser.add_argument("--inference-config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--baseline", type=Path, help="compare against a saved report; exit 2 for regressions")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--no-repeat-ngram-size", type=int)
    parser.add_argument(
        "--output", type=Path,
        help="also write the JSON result atomically (for example reports/generation_quality.json)",
    )
    args = parser.parse_args()
    if args.threads < 1 or args.max_tokens < 1:
        parser.error("threads and max-tokens must be positive")
    if args.output and args.output.exists():
        parser.error("output already exists; choose a new path to preserve evaluation evidence")
    torch.set_num_threads(args.threads)
    for label, path in (
        ("benchmark cases", args.cases),
        ("model configuration", args.model_config),
        ("inference configuration", args.inference_config),
        ("tokenizer", args.tokenizer),
        ("checkpoint", args.checkpoint),
    ):
        exists = path.is_dir() if label == "tokenizer" else path.is_file()
        if not exists:
            message = f"{label} not found: {path}"
            if label == "checkpoint":
                available = sorted(Path("checkpoints").glob("**/*.pt"))
                if available:
                    message += "\nAvailable checkpoints:\n  " + "\n  ".join(map(str, available))
                message += "\nFor an external checkpoint, mount its drive and select its matching tokenizer."
            parser.error(message)
    cases = []
    with args.cases.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            cases.append(BenchmarkCase(
                item["category"], item["prompt"], tuple(item["expected"]),
                tuple(item.get("forbidden", ())), item.get("match", "contains"),
            ))
    if not cases or len({(c.category, c.prompt) for c in cases}) != len(cases):
        parser.error("cases must be nonempty and unique")
    device = resolve_device(args.device)
    tokenizer = Tokenizer.load(args.tokenizer)
    model_config = load_yaml(args.model_config)
    inference_config = load_yaml(args.inference_config)
    try:
        model_config = adapt_config_to_tokenizer(model_config, tokenizer)
    except ValueError as error:
        parser.error(str(error))
    model = MiniGPT.from_config(model_config, device="cpu")
    checkpoint_info = load_checkpoint(
        args.checkpoint, model, use_ema=args.weights == "ema", restore_rng=False,
        **checkpoint_tokenizer_options(tokenizer, allow_extension=False),
    )
    generator = Generator(model, tokenizer, device=device)
    response_format = str(inference_config.get("response_format", "plain"))
    system_prompt = format_system_prompt(
        str(inference_config.get(
            "system_prompt", "You are Gopi, a helpful, honest, and friendly AI assistant."
        )),
        response_format,
        include_safety_instruction=bool(inference_config.get("embed_safety_instruction", True)),
    )
    repetition_penalty = (
        args.repetition_penalty
        if args.repetition_penalty is not None
        else float(inference_config.get("repetition_penalty", 1.1))
    )
    if repetition_penalty <= 0:
        parser.error("repetition penalty must be positive")
    no_repeat_ngram_size = (
        args.no_repeat_ngram_size
        if args.no_repeat_ngram_size is not None
        else int(inference_config.get("no_repeat_ngram_size", 3))
    )
    if no_repeat_ngram_size < 0:
        parser.error("no-repeat n-gram size must be non-negative")

    protocol = {
        "scorer_version": 2,
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "system_prompt": system_prompt,
        "max_tokens": args.max_tokens,
        "temperature": 0.0, "top_k": 0,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "min_tokens": 1, "tools": False,
    }
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    if baseline is not None and baseline.get("protocol") != protocol:
        parser.error("baseline protocol differs; rerun the baseline with the same settings")

    from datasets.preprocessor import format_messages

    scored = []
    details = []
    for case in cases:
        started = time.perf_counter()
        rendered_prompt = format_messages(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": case.prompt},
            ],
            add_generation_prompt=True,
        )
        result = generator.generate(
            rendered_prompt,
            max_tokens=args.max_tokens,
            temperature=0.0,
            top_k=0,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            allow_special_tokens=True,
        )
        score = score_answer(result.text, case)
        scored.append((case, score))
        details.append({"category": case.category, "prompt": case.prompt, "answer": result.text, "score": score,
                        "completion_tokens": len(result.token_ids), "prompt_tokens": result.prompt_tokens,
                        "finish_reason": result.finish_reason, "seconds": time.perf_counter() - started})
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint), "step": checkpoint_info["step"],
        "checkpoint_size_bytes": args.checkpoint.stat().st_size,
        "checkpoint_mtime_ns": args.checkpoint.stat().st_mtime_ns,
        "tokenizer_fingerprint": tokenizer.fingerprint, "tokenizer": str(args.tokenizer),
        "weights_requested": args.weights, "ema_applied": checkpoint_info["ema_applied"],
        "model_config": model_config, "device": str(device), "torch_version": str(torch.__version__),
        "protocol": protocol, "summary": summarize_scores(scored), "results": details,
        "note": "Small diagnostic suite, not standardized benchmark accuracy. Audit training overlap separately.",
    }
    if baseline is not None:
        report["comparison"] = compare_reports(baseline, report)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
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
    if baseline is not None and not report["comparison"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
