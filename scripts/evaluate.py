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
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.gpu.yaml"))
    parser.add_argument("--training-config", type=Path, default=Path("configs/training.cpu.yaml"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latest/model.pt"))
    parser.add_argument("--dataset", type=Path, action="append")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches", type=int, help="limit evaluation batches for a smoke test")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(
            f"checkpoint not found: {args.checkpoint}. "
            "Train one first with `python scripts/train.py --epochs 1`, or pass "
            "an existing file with `--checkpoint PATH`."
        )
    model_config, config = load_yaml(args.model_config), load_yaml(args.training_config)
    tokenizer = Tokenizer.load(args.tokenizer)
    if tokenizer.vocab_size != int(model_config["vocab_size"]):
        parser.error("tokenizer vocabulary does not match model vocab_size")
    max_seq_len = int(config.get("max_sequence_length", 0))
    max_pos = int(model_config.get("max_position", 0))
    if max_seq_len > max_pos:
        parser.error(
            f"training/evaluation max_sequence_length ({max_seq_len}) exceeds model max_position ({max_pos}). "
            f"Use a matching config (e.g. max_sequence_length <= {max_pos}) or a model config with max_position >= {max_seq_len}."
        )
    device = resolve_device(args.device)
    model = MiniGPT.from_config(model_config, device=device)
    load_checkpoint(args.checkpoint, model, map_location=device, use_ema=True)
    paths = args.dataset or config.get("validation_files") or config["train_files"]
    loader = build_loader(paths, tokenizer, config, shuffle=False)
    metrics = Evaluator(model, loss_fn=CausalLanguageModelLoss.from_config(config), device=device).evaluate(
        loader, max_batches=args.max_batches
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
