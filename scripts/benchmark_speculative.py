"""Benchmark greedy speculative decoding against ordinary greedy generation."""
from __future__ import annotations

try:
    from scripts._bootstrap import PROJECT_ROOT  # noqa: F401
except ModuleNotFoundError:
    from _bootstrap import PROJECT_ROOT  # noqa: F401

import argparse
import json
import statistics
import time
from pathlib import Path

from inference.generator import Generator
from model.gpt import MiniGPT
from model.vocabulary import adapt_config_to_tokenizer, checkpoint_tokenizer_options
from tokenizer.encoder import Tokenizer
from training.checkpoint import load_checkpoint
from utils.config import load_yaml
from utils.device import resolve_device


def _load(config_path: Path, checkpoint: Path, tokenizer: Tokenizer, device):
    config = adapt_config_to_tokenizer(load_yaml(config_path), tokenizer)
    model = MiniGPT.from_config(config, device="cpu")
    load_checkpoint(checkpoint, model, use_ema=True, restore_rng=False,
                    **checkpoint_tokenizer_options(tokenizer, allow_extension=False))
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--draft-checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.gpu.yaml"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--draft-tokens", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.max_tokens < 1 or args.draft_tokens < 1:
        parser.error("repeats, max-tokens and draft-tokens must be positive")
    device = resolve_device(args.device)
    tokenizer = Tokenizer.load(args.tokenizer)
    target = _load(args.model_config, args.target_checkpoint, tokenizer, device)
    draft = _load(args.model_config, args.draft_checkpoint, tokenizer, device)
    generator = Generator(target, tokenizer, device=device)
    greedy_times, speculative_times, acceptance = [], [], []
    greedy_result = speculative_result = None
    for _ in range(args.repeats):
        started = time.perf_counter()
        greedy_result = generator.generate(args.prompt, max_tokens=args.max_tokens, temperature=0.0, top_k=0)
        greedy_times.append(time.perf_counter() - started)
        started = time.perf_counter()
        speculative_result = generator.generate_speculative(
            args.prompt, draft, max_tokens=args.max_tokens, draft_tokens=args.draft_tokens
        )
        speculative_times.append(time.perf_counter() - started)
        acceptance.append(generator.last_speculative_stats["acceptance_rate"])
        if speculative_result.text != greedy_result.text:
            raise RuntimeError("speculative output diverged from deterministic greedy baseline")
    report = {
        "benchmark": "SPC-001 greedy speculative decoding",
        "protocol": {"max_tokens": args.max_tokens, "draft_tokens": args.draft_tokens, "repeats": args.repeats, "temperature": 0.0},
        "target_checkpoint": str(args.target_checkpoint),
        "draft_checkpoint": str(args.draft_checkpoint),
        "device": str(device),
        "greedy_median_seconds": statistics.median(greedy_times),
        "speculative_median_seconds": statistics.median(speculative_times),
        "speedup": statistics.median(greedy_times) / statistics.median(speculative_times),
        "mean_acceptance_rate": statistics.mean(acceptance),
        "output_match": True,
        "completion_tokens": len(greedy_result.token_ids),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
