"""Evaluate a frozen-backbone multimodal projector on an image-text JSONL split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model.gpt import MiniGPT
from multimodal.model import VisionLanguageModel, collate_image_text_sft
from multimodal.sft import ImageTextSFTDataset, evaluate_projector_batch
from tokenizer.encoder import Tokenizer
from training.checkpoint import load_checkpoint
from utils.config import load_yaml
from vision.encoder import VisionEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vision-config", type=Path, default=Path("configs/vision/model.small.yaml"))
    parser.add_argument("--language-config", type=Path, default=Path("configs/model.gpu.yaml"))
    parser.add_argument("--language-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer-finetuning"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_batches < 0:
        parser.error("batch-size must be positive and max-batches non-negative")
    device = torch.device(args.device)
    tokenizer = Tokenizer.load(args.tokenizer)
    language_config = load_yaml(args.language_config)
    language = MiniGPT.from_config(language_config, device=device)
    load_checkpoint(args.language_checkpoint, language, map_location=device, restore_rng=False)
    vision_config = load_yaml(args.vision_config)
    vision = VisionEncoder(**{key: value for key, value in vision_config.items() if key in {
        "image_size", "patch_size", "channels", "hidden_size", "layers", "heads",
        "ffn_hidden_size", "dropout", "initializer_range", "strict_image_size", "pool_type"
    }})
    model = VisionLanguageModel(vision, language, visual_tokens=16, freeze_vision=True, freeze_language=True).to(device).eval()
    dataset = ImageTextSFTDataset(args.manifest, tokenizer, image_size=vision.patch_embedding.image_size)
    totals = {"tokens": 0.0, "correct": 0.0, "loss": 0.0}
    batches = 0
    for start in range(0, len(dataset), args.batch_size):
        if args.max_batches and batches >= args.max_batches:
            break
        examples = [dataset[index] for index in range(start, min(start + args.batch_size, len(dataset)))]
        batch = collate_image_text_sft(examples)
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        metrics = evaluate_projector_batch(model, batch)
        tokens = metrics["tokens"]
        totals["tokens"] += tokens
        totals["correct"] += metrics["token_accuracy"] * tokens
        totals["loss"] += torch.log(torch.tensor(metrics["perplexity"])).item() * tokens
        batches += 1
    if not batches or not totals["tokens"]:
        parser.error("evaluation split produced no supervised response tokens")
    report = {
        "task": "VIS-002",
        "batches": batches,
        "tokens": totals["tokens"],
        "token_accuracy": totals["correct"] / totals["tokens"],
        "perplexity": float(torch.exp(torch.tensor(totals["loss"] / totals["tokens"]))),
        "frozen_vision": True,
        "frozen_language": True,
        "manifest": str(args.manifest),
        "note": "Evaluation evidence is diagnostic; corpus activation still requires DATA-003 governance artifacts.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
