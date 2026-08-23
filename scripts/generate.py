"""Generate text from a trained Gopi checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

from inference.generator import Generator
from model.gpt import MiniGPT
from tokenizer.encoder import Tokenizer
from training.checkpoint import load_checkpoint
from utils.config import load_yaml
from utils.device import resolve_device
from utils.logger import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.cpu.yaml"))
    parser.add_argument("--inference-config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--raw", action="store_true", help="Do not wrap prompt in chat template")
    args = parser.parse_args()

    configure_logging()
    model_config = load_yaml(args.model_config)
    inference_config = load_yaml(args.inference_config)
    tokenizer = Tokenizer.load(args.tokenizer)
    if tokenizer.vocab_size != int(model_config["vocab_size"]):
        parser.error("tokenizer vocabulary does not match model vocab_size")

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        for candidate in (
            Path("checkpoints/finetuning/best.pt"),
            Path("checkpoints/latest/model.pt"),
            Path("checkpoints/latest/best.pt"),
            Path("checkpoints/pretraining/best.pt"),
        ):
            if candidate.exists():
                checkpoint_path = candidate
                break
        if checkpoint_path is None:
            checkpoint_path = Path("checkpoints/latest/model.pt")

    print(f"Loading checkpoint: {checkpoint_path}")
    device = resolve_device(args.device)
    model = MiniGPT.from_config(model_config, device=device)
    load_checkpoint(checkpoint_path, model, map_location=device)

    system_prompt = str(inference_config.get("system_prompt", "You are Gopi, a helpful, honest, and friendly AI assistant."))
    if args.raw:
        rendered_prompt = args.prompt
    else:
        from datasets.preprocessor import format_messages
        rendered_prompt = format_messages(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": args.prompt}],
            add_generation_prompt=True,
        )

    print(f"Rendered Prompt: {repr(rendered_prompt)}")
    prompt_token_ids = tokenizer.encode(rendered_prompt, add_bos=True, allowed_special="all")
    print(f"Prompt Token IDs (first 30): {prompt_token_ids[:30]}")

    generator = Generator(model, tokenizer, device=device)
    result = generator.generate(
        rendered_prompt,
        max_tokens=args.max_tokens or int(inference_config.get("max_tokens", 512)),
        temperature=(args.temperature if args.temperature is not None else float(inference_config.get("temperature", 0.8))),
        top_k=int(inference_config.get("top_k", 40)),
        top_p=float(inference_config.get("top_p", 1.0)),
        repetition_penalty=float(inference_config.get("repetition_penalty", 1.0)),
        seed=args.seed,
        allow_special_tokens=True,
    )
    print("\nGenerated Output:")
    print(result.text)


if __name__ == "__main__":
    main()
