"""Chat interactively with a fine-tuned Gopi checkpoint."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

from dotenv import load_dotenv

from inference.context import ConversationMemory, format_system_prompt
from inference.generator import Generator
from inference.web_search import SearchResult, build_search_prompt, search_brave, search_searxng
from model.gpt import MiniGPT
from tokenizer.encoder import Tokenizer
from training.checkpoint import load_checkpoint
from utils.config import load_yaml
from utils.device import resolve_device

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.v2.gpu.yaml"))
    parser.add_argument("--inference-config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer-v2"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/v2-finetuning/best.pt"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--response-format", choices=("plain", "markdown"))
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}; finish v2 fine-tuning first")

    model_config = load_yaml(args.model_config)
    inference_config = load_yaml(args.inference_config)
    tokenizer = Tokenizer.load(args.tokenizer)
    device = resolve_device(args.device)
    model = MiniGPT.from_config(model_config, device=device)
    load_checkpoint(
        args.checkpoint, model, map_location=device,
        expected_tokenizer_fingerprint=tokenizer.fingerprint,
    )
    generator = Generator(model, tokenizer, device=device)

    max_tokens = args.max_tokens or int(inference_config.get("max_tokens", 128))
    configured_context = int(inference_config.get("context_memory", {}).get("max_tokens", 1536))
    active_system_prompt = str(inference_config.get("system_prompt", "You are Gopi, a helpful AI assistant."))
    response_format = (
        args.response_format
        or os.getenv("GOPI_RESPONSE_FORMAT")
        or str(inference_config.get("response_format", "plain"))
    ).lower()
    try:
        formatted_system_prompt = format_system_prompt(active_system_prompt, response_format)
    except ValueError as error:
        parser.error(str(error))
    memory = ConversationMemory(
        tokenizer,
        max_tokens=min(configured_context, generator.max_positions - 1),
        system_prompt=formatted_system_prompt,
    )

    search_config = inference_config.get("web_search", {})
    search_provider = os.getenv("GOPI_SEARCH_PROVIDER", str(search_config.get("provider", "searxng"))).lower()
    search_api_key = os.getenv("GOPI_SEARCH_API_KEY", "")

    print("Gopi chat ready. Use /help to list commands.")
    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not message:
            continue
        search_results: list[SearchResult] = []
        if message.lower() in {"/quit", "/exit"}:
            break
        if message.lower() == "/clear":
            memory.clear()
            print("Conversation cleared.")
            continue
        if message.lower() == "/help":
            print(
                "Commands:\n"
                "  /user <message>   Send an explicit user message\n"
                "  /system <prompt>  Replace the current system prompt\n"
                "  /search <query>   Search the web, then answer with sources\n"
                "  /format <mode>    Set response format: plain or markdown\n"
                "  /clear            Clear conversation history\n"
                "  /quit             Exit"
            )
            continue
        if message.lower() == "/user" or message.lower().startswith("/user "):
            message = message[5:].strip()
            if not message:
                print("Usage: /user <message>")
                continue
        elif message.lower() == "/system" or message.lower().startswith("/system "):
            prompt = message[7:].strip()
            if not prompt:
                print("Usage: /system <prompt>")
                continue
            try:
                active_system_prompt = prompt
                memory.set_system_prompt(format_system_prompt(active_system_prompt, response_format))
            except ValueError as error:
                print(f"Could not set system prompt: {error}")
            else:
                print("System prompt updated.")
            continue
        elif message.lower() == "/format" or message.lower().startswith("/format "):
            requested_format = message[7:].strip().lower()
            if requested_format not in {"plain", "markdown"}:
                print("Usage: /format <plain|markdown>")
                continue
            response_format = requested_format
            memory.set_system_prompt(format_system_prompt(active_system_prompt, response_format))
            print(f"Response format set to {response_format}.")
            continue
        elif message.lower() == "/search" or message.lower().startswith("/search "):
            query = message[7:].strip()
            if not query:
                print("Usage: /search <query>")
                continue
            try:
                max_results = int(search_config.get("max_results", 5))
                timeout = float(search_config.get("timeout_seconds", 10.0))
                if search_provider == "searxng":
                    search_results = search_searxng(
                        query,
                        max_results=max_results,
                        timeout=timeout,
                        endpoint=os.getenv(
                            "GOPI_SEARXNG_URL",
                            str(search_config.get("searxng_endpoint", "http://localhost:8080/search")),
                        ),
                    )
                elif search_provider == "brave":
                    search_results = search_brave(
                        query,
                        search_api_key,
                        max_results=max_results,
                        timeout=timeout,
                        endpoint=str(search_config.get("brave_endpoint", "https://api.search.brave.com/res/v1/web/search")),
                    )
                else:
                    raise ValueError(f"unsupported search provider: {search_provider}")
            except (HTTPError, URLError, TimeoutError, ValueError) as error:
                print(f"Search failed: {error}")
                continue
            if not search_results:
                print("No web results found.")
                continue
            message = build_search_prompt(
                query,
                search_results,
                description_char_limit=int(search_config.get("description_char_limit", 200)),
            )
        result = generator.generate_chat(
            memory,
            message,
            max_tokens=max_tokens,
            temperature=(
                args.temperature if args.temperature is not None
                else float(search_config.get("temperature", 0.2)) if search_results
                else float(inference_config.get("temperature", 0.7))
            ),
            top_k=int(search_config.get("top_k", 20)) if search_results else int(inference_config.get("top_k", 40)),
            top_p=float(inference_config.get("top_p", 0.9)),
            repetition_penalty=float(inference_config.get("repetition_penalty", 1.1)),
        )
        print(f"Gopi: {result.text.strip()}")
        if search_results:
            print("Sources:")
            for index, item in enumerate(search_results, start=1):
                print(f"  [{index}] {item.title}: {item.url}")


if __name__ == "__main__":
    main()
