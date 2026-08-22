"""Evaluate a Gopi checkpoint on tokenized conversation data."""

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
from tokenizer.encoder import Tokenizer
from training.checkpoint import load_checkpoint
from training.data import build_loader
from training.evaluator import Evaluator
from utils.config import load_yaml
from utils.device import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--training-config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latest/model.pt"))
    parser.add_argument("--dataset", type=Path, action="append")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    model_config, config = load_yaml(args.model_config), load_yaml(args.training_config)
    tokenizer = Tokenizer.load(args.tokenizer)
    device = resolve_device(args.device)
    model = MiniGPT.from_config(model_config, device=device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    paths = args.dataset or config.get("validation_files") or config["train_files"]
    loader = build_loader(paths, tokenizer, config, shuffle=False)
    metrics = Evaluator(model, loss_fn=CausalLanguageModelLoss.from_config(config), device=device).evaluate(loader)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
