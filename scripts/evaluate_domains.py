"""Evaluate validation loss independently for each configured capability domain."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

from model.gpt import MiniGPT
from model.loss import CausalLanguageModelLoss
from model.vocabulary import adapt_config_to_tokenizer, checkpoint_tokenizer_options
from tokenizer.encoder import Tokenizer
from training.checkpoint import load_checkpoint
from training.data import build_loader
from training.evaluator import Evaluator
from utils.config import load_yaml
from utils.device import resolve_device


def aggregate_domain_metrics(results: dict[str, dict[str, float | int]]) -> dict[str, float | int]:
    """Combine domain metrics without allowing small domains to dominate."""
    total_tokens = sum(int(metrics["tokens"]) for metrics in results.values())
    aggregate: dict[str, float | int] = {}
    for key in ("loss", "cross_entropy", "z_loss"):
        aggregate[key] = sum(
            float(metrics[key]) * int(metrics["tokens"]) for metrics in results.values()
        ) / max(total_tokens, 1)
    aggregate["perplexity"] = math.exp(min(float(aggregate["cross_entropy"]), 80.0))
    aggregate["tokens"] = total_tokens
    aggregate["batches"] = sum(int(metrics["batches"]) for metrics in results.values())
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", type=Path, default=Path("configs/evaluation.v2.domains.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.v2.gpu.yaml"))
    parser.add_argument("--training-config", type=Path, default=Path("configs/finetuning.v2.gpu.yaml"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer-v2"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches", type=int, help="limit batches independently per domain")
    args = parser.parse_args()

    domain_config = load_yaml(args.domains)
    domains = domain_config.get("domains")
    if not isinstance(domains, dict) or not domains:
        parser.error("domain configuration must contain a non-empty domains mapping")
    training_config = load_yaml(args.training_config)
    tokenizer = Tokenizer.load(args.tokenizer)
    try:
        model_config = adapt_config_to_tokenizer(load_yaml(args.model_config), tokenizer)
    except ValueError as error:
        parser.error(str(error))
    device = resolve_device(args.device)
    model = MiniGPT.from_config(model_config, device=device)
    load_checkpoint(
        args.checkpoint, model, map_location=device, use_ema=True,
        **checkpoint_tokenizer_options(tokenizer),
    )
    evaluator = Evaluator(
        model,
        loss_fn=CausalLanguageModelLoss.from_config(training_config),
        device=device,
        mixed_precision=str(training_config.get("mixed_precision", "none")),
    )

    results = {}
    for domain, paths in domains.items():
        if not isinstance(paths, list) or not paths:
            parser.error(f"domain {domain!r} must contain dataset paths")
        loader = build_loader(paths, tokenizer, training_config, shuffle=False)
        results[str(domain)] = evaluator.evaluate(loader, max_batches=args.max_batches)

    aggregate = aggregate_domain_metrics(results)
    print(json.dumps({"aggregate": aggregate, "domains": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
