"""Prepare a deterministic UltraChat subset for supervised fine-tuning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Avoid resolving the standard-library ``tokenize`` module to scripts/tokenize.py
# when this file is executed directly.
script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

import pyarrow.parquet as pq


def valid_messages(messages: list[dict[str, str]] | None) -> bool:
    if not messages or len(messages) < 2:
        return False
    expected_role = "user"
    for message in messages:
        if message.get("role") != expected_role or not message.get("content", "").strip():
            return False
        expected_role = "assistant" if expected_role == "user" else "user"
    return messages[-1]["role"] == "assistant"


def write_subset(
    source: Path,
    destination: Path,
    limit: int,
    offset: int = 0,
    bot_name: str = "Gopi",
) -> int:
    written = 0
    eligible = 0
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8") as output:
        parquet = pq.ParquetFile(source)
        for batch in parquet.iter_batches(columns=["prompt_id", "messages"]):
            for row in batch.to_pylist():
                messages = row["messages"]
                if not valid_messages(messages):
                    continue
                if eligible < offset:
                    eligible += 1
                    continue
                record = {
                    "id": row["prompt_id"],
                    "bot_name": bot_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                f"You are {bot_name}, a helpful, honest, and friendly AI assistant."
                            ),
                        },
                        *[
                            {
                                "role": message["role"],
                                "content": message["content"].strip(),
                            }
                            for message in messages
                        ],
                    ],
                    "source": "HuggingFaceH4/ultrachat_200k",
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                eligible += 1
                if written == limit:
                    return written
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/ultrachat_200k"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/ultrachat_200k")
    )
    parser.add_argument("--train-size", type=int, default=20_000)
    parser.add_argument("--validation-size", type=int, default=2_000)
    parser.add_argument("--test-size", type=int, default=2_000)
    parser.add_argument("--bot-name", default="Gopi")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_source = next(args.raw_dir.glob("train_sft-*.parquet"))
    test_source = next(args.raw_dir.glob("test_sft-*.parquet"))

    counts = {
        "train": write_subset(
            train_source, args.output_dir / "train.jsonl", args.train_size, bot_name=args.bot_name
        ),
        "validation": write_subset(
            test_source,
            args.output_dir / "validation.jsonl",
            args.validation_size,
            bot_name=args.bot_name,
        ),
        "test": write_subset(
            test_source,
            args.output_dir / "test.jsonl",
            args.test_size,
            offset=args.validation_size,
            bot_name=args.bot_name,
        ),
    }
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
