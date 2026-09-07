"""Export an existing MiniGPT training checkpoint as a local inference bundle."""
from __future__ import annotations

import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
sys.path[:] = [entry for entry in sys.path if str(Path(entry or '.').resolve()) != script_directory]

import argparse

from inference.pretrained import MiniGPTBackend
from utils.config import load_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inference-config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--raw-weights", action="store_true", help="use model weights instead of available EMA weights")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    config = load_yaml(args.inference_config)
    keys = {"max_tokens", "temperature", "top_k", "top_p", "min_p", "min_tokens",
            "repetition_penalty", "no_repeat_ngram_size", "stop"}
    backend = MiniGPTBackend.from_checkpoint(
        args.checkpoint, model_config=load_yaml(args.model_config), tokenizer=args.tokenizer,
        device="cpu", use_ema=not args.raw_weights,
        generation_config={key: config[key] for key in keys if key in config},
    )
    print(backend.save_pretrained(args.output))


if __name__ == "__main__":
    main()
