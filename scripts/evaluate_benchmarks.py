"""Run deterministic held-out generation checks grouped by capability."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

from evaluation.benchmarks import BenchmarkCase, score_answer, summarize_scores
from inference.generator import Generator
from model.gpt import MiniGPT
from model.vocabulary import adapt_config_to_tokenizer, checkpoint_tokenizer_options
from tokenizer.encoder import Tokenizer
from training.checkpoint import load_checkpoint
from utils.config import load_yaml
from utils.device import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("configs/evaluation.core.jsonl"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.v2.gpu.yaml"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer-v2"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    for label, path in (
        ("benchmark cases", args.cases),
        ("model configuration", args.model_config),
        ("tokenizer", args.tokenizer),
        ("checkpoint", args.checkpoint),
    ):
        exists = path.is_dir() if label == "tokenizer" else path.is_file()
        if not exists:
            message = f"{label} not found: {path}"
            if label == "checkpoint":
                available = sorted(Path("checkpoints").glob("**/*.pt"))
                if available:
                    message += "\nAvailable checkpoints:\n  " + "\n  ".join(map(str, available))
                message += (
                    "\nA v2 pretraining checkpoint must be fine-tuned before running "
                    "assistant capability benchmarks."
                )
            parser.error(message)
    cases = []
    with args.cases.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            cases.append(BenchmarkCase(
                item["category"], item["prompt"], tuple(item["expected"]), tuple(item.get("forbidden", ()))
            ))
    device = resolve_device(args.device)
    tokenizer = Tokenizer.load(args.tokenizer)
    model_config = load_yaml(args.model_config)
    try:
        model_config = adapt_config_to_tokenizer(model_config, tokenizer)
    except ValueError as error:
        parser.error(str(error))
    model = MiniGPT.from_config(model_config, device=device)
    load_checkpoint(
        args.checkpoint, model, map_location=device, use_ema=True,
        **checkpoint_tokenizer_options(tokenizer),
    )
    generator = Generator(model, tokenizer, device=device)
    scored = []
    details = []
    for case in cases:
        result = generator.generate(case.prompt, max_tokens=args.max_tokens, temperature=0.0, top_k=0)
        score = score_answer(result.text, case)
        scored.append((case, score))
        details.append({"category": case.category, "prompt": case.prompt, "answer": result.text, "score": score})
    print(json.dumps(
        {"summary": summarize_scores(scored), "results": details},
        indent=2, ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
