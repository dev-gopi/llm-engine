"""Evaluate validation loss independently for each configured capability domain."""

from __future__ import annotations

import argparse
import json
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
from training.evaluator import Evaluator, aggregate_domain_metrics
from utils.config import load_yaml
from utils.device import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", type=Path, default=Path("configs/evaluation.domains.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.gpu.yaml"))
    parser.add_argument("--training-config", type=Path, default=Path("configs/finetuning.gpu.yaml"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches", type=int, help="limit batches independently per domain")
    args = parser.parse_args()

    domain_config = load_yaml(args.domains)
    domains = domain_config.get("domains")
    if not isinstance(domains, dict) or not domains:
        parser.error("domain configuration must contain a non-empty domains mapping")
    weights = domain_config.get("weights")
    if not isinstance(weights, dict):
        parser.error("domain configuration must contain an explicit weights mapping")
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

    try:
        aggregate = aggregate_domain_metrics(results, weights)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({"aggregate": aggregate, "domains": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
