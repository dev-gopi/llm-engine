"""Evaluate and calibrate a trained pairwise reward model on held-out preferences."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
repository_root = str(Path(__file__).resolve().parents[1])
src_root = str(Path(__file__).resolve().parents[1] / "src")
sys.path[:] = [entry for entry in sys.path if str(Path(entry or ".").resolve()) not in {script_directory, repository_root, src_root}]
sys.path[:0] = [src_root, repository_root]

import torch

from local_dataset.loader import iter_records
from model.gpt import MiniGPT
from model.vocabulary import adapt_config_to_tokenizer, checkpoint_tokenizer_options
from post_training.reward_model import RewardModel
from post_training.reward_scoring import RewardScorer, fit_reward_calibration
from tokenizer.encoder import Tokenizer
from training.checkpoint import load_checkpoint
from utils.config import load_yaml
from utils.device import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.gpu.yaml"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer-finetuning"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    tokenizer = Tokenizer.load(args.tokenizer)
    config = adapt_config_to_tokenizer(load_yaml(args.model_config), tokenizer)
    device = resolve_device(args.device)
    model = RewardModel(MiniGPT.from_config(config, device=device), int(config["hidden_size"])).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device, restore_rng=False, **checkpoint_tokenizer_options(tokenizer))
    scorer = RewardScorer(model, tokenizer, max_length=args.max_sequence_length)
    pairs = []
    for path in args.input:
        for record in iter_records(path):
            prompt, chosen, rejected = record.get("prompt"), record.get("chosen"), record.get("rejected")
            if all(isinstance(value, str) and value.strip() for value in (prompt, chosen, rejected)):
                pairs.append((prompt.strip(), chosen.strip(), rejected.strip()))
    if not pairs:
        parser.error("no usable chosen/rejected calibration pairs")
    chosen_scores = scorer.score_many(((p, c) for p, c, _ in pairs), batch_size=args.batch_size)
    rejected_scores = scorer.score_many(((p, r) for p, _, r in pairs), batch_size=args.batch_size)
    calibration, metrics = fit_reward_calibration(chosen_scores, rejected_scores, clip=args.clip)
    calibration.save(args.output, metadata={
        "checkpoint": str(args.checkpoint),
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "metrics": metrics,
    })
    print(json.dumps({"calibration": str(args.output), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
