"""Generate text directly from an exported SafeTensors model bundle."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path[:] = [
    entry for entry in sys.path
    if os.path.realpath(entry or ".") != script_directory
]

from safetensors.torch import load_model

from inference.generator import Generator
from model.gpt import MiniGPT
from model.vocabulary import adapt_config_to_tokenizer
from tokenizer.encoder import Tokenizer
from utils.config import load_yaml
from utils.device import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="raw text prompt to continue")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("exports/v2-pretraining/gopi-v2.safetensors"),
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("exports/v2-pretraining/model.yaml"),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("exports/v2-pretraining/tokenizer"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for label, path, directory in (
        ("model", args.model, False),
        ("model config", args.model_config, False),
        ("tokenizer", args.tokenizer, True),
    ):
        exists = path.is_dir() if directory else path.is_file()
        if not exists:
            parser.error(f"{label} not found: {path}")

    tokenizer = Tokenizer.load(args.tokenizer)
    try:
        model_config = adapt_config_to_tokenizer(
            load_yaml(args.model_config), tokenizer,
        )
    except ValueError as error:
        parser.error(str(error))

    device = resolve_device(args.device)
    model = MiniGPT.from_config(model_config, device="cpu")
    load_model(model, args.model)
    model.to(device).eval()

    generator = Generator(model, tokenizer, device=device)
    result = generator.generate(
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        allow_special_tokens=True,
    )
    print(result.text)


if __name__ == "__main__":
    main()
