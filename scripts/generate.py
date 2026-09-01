"""Generate text from a trained Gopi checkpoint."""

from __future__ import annotations

import os
import sys

script_directory = os.path.dirname(os.path.realpath(__file__))
# The script directory can occur more than once (for example through
# PYTHONPATH). Remove every copy so scripts/tokenize.py cannot shadow the
# standard-library tokenize module imported by PyTorch.
sys.path[:] = [entry for entry in sys.path if os.path.realpath(entry or ".") != script_directory]

import argparse
import asyncio
from pathlib import Path

from inference.context import format_system_prompt
from inference.generator import Generator
from inference.web_search import build_search_prompt, format_sources, search_brave, search_searxng
from model.gpt import MiniGPT
from model.vocabulary import adapt_config_to_tokenizer, checkpoint_tokenizer_options
from serving.backend import ConfiguredModelBackend
from serving.schemas import GenerateRequest
from tokenizer.encoder import Tokenizer
from training.checkpoint import load_checkpoint
from utils.config import load_yaml
from utils.device import resolve_device
from utils.logger import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", help="text to generate a response for")
    parser.add_argument(
        "--prompt", dest="prompt_option",
        help="text to generate a response for (alternative to the positional prompt)",
    )
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.v2.gpu.yaml"))
    parser.add_argument("--inference-config", type=Path, default=Path("configs/inference.v2.yaml"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer-v2"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--max-tokens", "--max-new-tokens", dest="max_tokens", type=int,
        help="maximum number of tokens to generate",
    )
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--raw", action="store_true", help="Do not wrap prompt in chat template")
    parser.add_argument("--response-format", choices=("plain", "markdown"))
    parser.add_argument("--search", action="store_true", help="Search the web before generating")
    parser.add_argument("--mcp", action="store_true", help="Allow an allowlisted MCP tool call before generating")
    parser.add_argument("--mcp-server", help="Restrict MCP routing to one configured server")
    parser.add_argument("--mcp-config", type=Path, default=Path("configs/mcp.yaml"))
    args = parser.parse_args()
    if args.prompt is not None and args.prompt_option is not None:
        parser.error("provide the prompt either positionally or with --prompt, not both")
    args.prompt = args.prompt_option if args.prompt_option is not None else args.prompt
    if args.prompt is None:
        parser.error("a prompt is required (positionally or with --prompt)")

    configure_logging()
    model_config = load_yaml(args.model_config)
    inference_config = load_yaml(args.inference_config)
    tokenizer = Tokenizer.load(args.tokenizer)
    try:
        model_config = adapt_config_to_tokenizer(model_config, tokenizer)
    except ValueError as error:
        parser.error(str(error))

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        for candidate in (
            Path("checkpoints/v2-training/best.pt"),
            Path("checkpoints/v2-training/latest.pt"),
            Path("checkpoints/v2-pretraining/best.pt"),
            Path("checkpoints/v2-pretraining/latest.pt"),
        ):
            if candidate.exists():
                checkpoint_path = candidate
                break
        if checkpoint_path is None:
            checkpoint_path = Path("checkpoints/v2-pretraining/best.pt")

    print(f"Loading checkpoint: {checkpoint_path}")
    device = resolve_device(args.device)
    model = MiniGPT.from_config(model_config, device=device)
    load_checkpoint(
        checkpoint_path, model, map_location=device, use_ema=True,
        **checkpoint_tokenizer_options(tokenizer),
    )

    response_format = args.response_format or str(inference_config.get("response_format", "plain"))
    system_prompt = format_system_prompt(
        str(inference_config.get("system_prompt", "You are Gopi, a helpful, honest, and friendly AI assistant.")),
        response_format,
    )
    prompt = args.prompt
    search_results = []
    slash_search = prompt.strip().lower() == "/search" or prompt.strip().lower().startswith("/search ")
    if args.search or slash_search:
        query = prompt.strip()[7:].strip() if slash_search else prompt.strip()
        if not query:
            parser.error("search query cannot be empty")
        search_config = inference_config.get("web_search", {})
        provider = os.getenv("GOPI_SEARCH_PROVIDER", str(search_config.get("provider", "searxng"))).lower()
        common = {
            "max_results": int(search_config.get("max_results", 3)),
            "timeout": float(search_config.get("timeout_seconds", 10.0)),
        }
        if provider == "searxng":
            search_results = search_searxng(
                query,
                endpoint=os.getenv("GOPI_SEARXNG_URL", str(search_config.get("searxng_endpoint", "http://localhost:8080/search"))),
                **common,
            )
        elif provider == "brave":
            search_results = search_brave(
                query,
                os.getenv("GOPI_SEARCH_API_KEY", ""),
                endpoint=str(search_config.get("brave_endpoint", "https://api.search.brave.com/res/v1/web/search")),
                **common,
            )
        else:
            parser.error(f"unsupported search provider: {provider}")
        if not search_results:
            parser.error("no web results found")
        prompt = build_search_prompt(
            query,
            search_results,
            description_char_limit=int(search_config.get("description_char_limit", 200)),
        )

    if args.raw:
        rendered_prompt = prompt
    else:
        from datasets.preprocessor import format_messages
        rendered_prompt = format_messages(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            add_generation_prompt=True,
        )

    generator = Generator(model, tokenizer, device=device)
    if args.mcp:
        if not args.mcp_config.is_file():
            parser.error(f"MCP configuration not found: {args.mcp_config}")
        mcp_root = load_yaml(args.mcp_config)
        mcp_config = mcp_root.get("mcp", {})
        if not isinstance(mcp_config, dict):
            parser.error("mcp configuration must be a mapping")

        async def augment_with_mcp() -> str:
            backend = ConfiguredModelBackend(mcp=mcp_config)
            backend.generator = generator
            try:
                await backend._startup_mcp()
                request = GenerateRequest(
                    prompt=prompt,
                    mcp=True,
                    mcp_server=args.mcp_server,
                    seed=args.seed,
                )
                return await backend._augment_with_mcp(request, prompt)
            finally:
                await backend.shutdown()

        prompt = asyncio.run(augment_with_mcp())

        if args.raw:
            rendered_prompt = prompt
        else:
            rendered_prompt = format_messages(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                add_generation_prompt=True,
            )

    print(f"Rendered Prompt: {repr(rendered_prompt)}")
    prompt_token_ids = tokenizer.encode(rendered_prompt, add_bos=True, allowed_special="all")
    print(f"Prompt Token IDs (first 30): {prompt_token_ids[:30]}")

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
    if search_results:
        print(f"\n{format_sources(search_results)}")


if __name__ == "__main__":
    main()
