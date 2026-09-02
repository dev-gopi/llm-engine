"""Grow a checkpoint by appending vocabulary rows and identity-initialized layers."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path[:] = [
    entry for entry in sys.path
    if os.path.realpath(entry or ".") != script_directory
]

from model.gpt import MiniGPT
from model.vocabulary import adapt_config_to_tokenizer, checkpoint_tokenizer_options
from tokenizer.encoder import Tokenizer
from training.checkpoint import load_checkpoint, save_checkpoint
from training.model_growth import grow_model
from utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--source-model-config", type=Path,
        default=Path("configs/model.source.gpu.yaml"),
    )
    parser.add_argument("--target-model-config", type=Path, default=Path("configs/model.gpu.yaml"))
    parser.add_argument("--source-tokenizer", type=Path, default=Path("data/tokenizer-v2"))
    parser.add_argument("--target-tokenizer", type=Path, default=Path("data/tokenizer-v3"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/grown/init.pt"))
    parser.add_argument("--embedding-init", choices=("mean", "normal", "zero"), default="mean")
    parser.add_argument("--use-ema", action="store_true", help="grow EMA weights instead of live weights")
    args = parser.parse_args()

    for label, path, directory in (
        ("checkpoint", args.checkpoint, False),
        ("source model config", args.source_model_config, False),
        ("target model config", args.target_model_config, False),
        ("source tokenizer", args.source_tokenizer, True),
        ("target tokenizer", args.target_tokenizer, True),
    ):
        exists = path.is_dir() if directory else path.is_file()
        if not exists:
            parser.error(f"{label} not found: {path}")

    source_tokenizer = Tokenizer.load(args.source_tokenizer)
    target_tokenizer = Tokenizer.load(args.target_tokenizer)
    if source_tokenizer.fingerprint not in target_tokenizer.compatible_base_fingerprints:
        parser.error("target tokenizer is not a verified append-only extension of source tokenizer")

    try:
        source_config = adapt_config_to_tokenizer(
            load_yaml(args.source_model_config), source_tokenizer
        )
        target_config = adapt_config_to_tokenizer(
            load_yaml(args.target_model_config), target_tokenizer
        )
    except ValueError as error:
        parser.error(str(error))

    source = MiniGPT.from_config(source_config, device="cpu")
    load_checkpoint(
        args.checkpoint,
        source,
        map_location="cpu",
        use_ema=args.use_ema,
        restore_rng=False,
        **checkpoint_tokenizer_options(source_tokenizer),
    )
    target = MiniGPT.from_config(target_config, device="cpu")
    try:
        report = grow_model(source, target, embedding_init=args.embedding_init)
    except ValueError as error:
        parser.error(str(error))

    save_checkpoint(
        args.output,
        target,
        step=0,
        metadata={
            "model_config": target_config,
            "tokenizer_fingerprint": target_tokenizer.fingerprint,
            "grown_from": str(args.checkpoint),
            "growth": report.__dict__,
        },
    )
    print(json.dumps({"output": str(args.output), **report.__dict__}, indent=2))


if __name__ == "__main__":
    main()
