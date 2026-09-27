"""Score prompt/completion JSONL records with a trained reward model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
repository_root = str(Path(__file__).resolve().parents[1])
src_root = str(Path(__file__).resolve().parents[1] / "src")
sys.path[:] = [
    entry
    for entry in sys.path
    if str(Path(entry or ".").resolve())
    not in {script_directory, repository_root, src_root}
]
sys.path[:0] = [src_root, repository_root]

from local_dataset.loader import iter_records
from model.gpt import MiniGPT
from model.vocabulary import adapt_config_to_tokenizer, checkpoint_tokenizer_options
from post_training.reward_model import RewardModel
from post_training.reward_scoring import RewardCalibration, RewardScorer
from tokenizer.encoder import Tokenizer
from training.checkpoint import load_checkpoint
from utils.config import load_yaml
from utils.device import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/model.gpu.yaml")
    )
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("data/tokenizer-finetuning")
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--normalized", action="store_true")
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    tokenizer = Tokenizer.load(args.tokenizer)
    config = adapt_config_to_tokenizer(load_yaml(args.model_config), tokenizer)
    device = resolve_device(args.device)
    model = RewardModel(
        MiniGPT.from_config(config, device=device), int(config["hidden_size"])
    ).to(device)
    load_checkpoint(
        args.checkpoint,
        model,
        map_location=device,
        restore_rng=False,
        **checkpoint_tokenizer_options(tokenizer),
    )
    calibration = RewardCalibration.load(args.calibration) if args.calibration else None
    scorer = RewardScorer(
        model, tokenizer, max_length=args.max_sequence_length, calibration=calibration
    )
    records = []
    score_pairs = []
    for record in iter_records(args.input):
        prompt, completion = record.get("prompt"), record.get("completion")
        if (
            isinstance(prompt, str)
            and isinstance(completion, str)
            and prompt.strip()
            and completion.strip()
        ):
            records.append(dict(record))
            score_pairs.append((prompt, completion))
    if not records:
        parser.error("input has no usable prompt/completion records")
    scores = scorer.score_many(
        score_pairs, batch_size=args.batch_size, normalized=args.normalized
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record, score in zip(records, scores, strict=True):
            record["reward"] = score
            record["reward_normalized"] = bool(args.normalized)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
