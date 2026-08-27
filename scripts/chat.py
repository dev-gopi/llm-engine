"""Chat interactively with a fine-tuned Gopi checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

from inference.context import ConversationMemory
from inference.generator import Generator
from model.gpt import MiniGPT
from tokenizer.encoder import Tokenizer
from training.checkpoint import load_checkpoint
from utils.config import load_yaml
from utils.device import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.v2.gpu.yaml"))
    parser.add_argument("--inference-config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer-v2"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/v2-finetuning/best.pt"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}; finish v2 fine-tuning first")

    model_config = load_yaml(args.model_config)
    inference_config = load_yaml(args.inference_config)
    tokenizer = Tokenizer.load(args.tokenizer)
    device = resolve_device(args.device)
    model = MiniGPT.from_config(model_config, device=device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    generator = Generator(model, tokenizer, device=device)

    max_tokens = args.max_tokens or int(inference_config.get("max_tokens", 128))
    configured_context = int(inference_config.get("context_memory", {}).get("max_tokens", 1536))
    memory = ConversationMemory(
        tokenizer,
        max_tokens=min(configured_context, generator.max_positions - 1),
        system_prompt=str(inference_config.get("system_prompt", "You are Gopi, a helpful AI assistant.")),
    )

    print("Gopi chat ready. Use /clear to reset or /quit to exit.")
    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not message:
            continue
        if message.lower() in {"/quit", "/exit"}:
            break
        if message.lower() == "/clear":
            memory.clear()
            print("Conversation cleared.")
            continue
        result = generator.generate_chat(
            memory,
            message,
            max_tokens=max_tokens,
            temperature=args.temperature if args.temperature is not None else float(inference_config.get("temperature", 0.7)),
            top_k=int(inference_config.get("top_k", 40)),
            top_p=float(inference_config.get("top_p", 0.9)),
            repetition_penalty=float(inference_config.get("repetition_penalty", 1.1)),
        )
        print(f"Gopi: {result.text.strip()}")


if __name__ == "__main__":
    main()
