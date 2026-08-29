"""Train and inspect the project's byte-level BPE tokenizer."""

from __future__ import annotations

import sys
from pathlib import Path

# This file is named tokenize.py by design. Remove its directory from the import
# path so dependencies can still import Python's standard-library tokenize module.
script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

import argparse
import glob
import heapq
import json
from collections import deque
from collections.abc import Iterable, Iterator
from typing import Any

import pyarrow.parquet as pq
import yaml

from tokenizer.encoder import Tokenizer
from tokenizer.trainer import BPETokenizerTrainer


def extract_text(value: Any) -> Iterator[str]:
    """Yield textual training fields while ignoring IDs and metadata strings."""

    if isinstance(value, dict):
        if isinstance(value.get("messages"), list):
            for message in value["messages"]:
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    yield message["content"]
            return
        if isinstance(value.get("text"), str):
            yield value["text"]
            return
        if isinstance(value.get("utterance"), str):
            yield value["utterance"]
            return
        for key, nested in value.items():
            if key not in {"id", "prompt_id", "source", "bot_name", "domain"}:
                yield from extract_text(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from extract_text(nested)


def read_jsonl(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield from extract_text(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON in {path}:{line_number}") from error


def read_json(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as stream:
        yield from extract_text(json.load(stream))


def read_parquet(path: Path) -> Iterator[str]:
    parquet = pq.ParquetFile(path)
    columns = set(parquet.schema_arrow.names)
    selected = [name for name in ("text", "messages", "utterance") if name in columns]
    if not selected:
        raise ValueError(f"no supported text column in {path}")
    for batch in parquet.iter_batches(columns=selected, batch_size=2_048):
        for record in batch.to_pylist():
            yield from extract_text(record)


def read_text(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield line


READERS = {
    ".jsonl": read_jsonl,
    ".json": read_json,
    ".parquet": read_parquet,
    ".txt": read_text,
}


def _corpus_readers(patterns: Iterable[str]) -> list[Iterator[str]]:
    readers: list[Iterator[str]] = []
    for pattern in patterns:
        matches = sorted(Path(item) for item in glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"dataset pattern matched no files: {pattern}")

        def read_matches(paths: tuple[Path, ...]) -> Iterator[str]:
            # A glob is one logical source. Process its shards sequentially so
            # balanced sampling holds at most one open file per configured
            # source instead of one descriptor per shard.
            for path in paths:
                try:
                    reader = READERS[path.suffix.lower()]
                except KeyError as error:
                    raise ValueError(f"unsupported dataset format: {path}") from error
                print(f"Reading {path}", file=sys.stderr)
                yield from reader(path)

        for path in matches:
            try:
                READERS[path.suffix.lower()]
            except KeyError as error:
                raise ValueError(f"unsupported dataset format: {path}") from error
        readers.append(read_matches(tuple(matches)))
    if not readers:
        raise ValueError("no tokenizer training sources were configured")
    return readers


def iter_corpus(patterns: Iterable[str], *, sampling: str = "sequential") -> Iterator[str]:
    readers = _corpus_readers(patterns)
    if sampling == "sequential":
        for reader in readers:
            yield from reader
        return
    if sampling == "round_robin":
        # Interleave one extracted text field from each source.
        pending = deque(readers)
        while pending:
            reader = pending.popleft()
            try:
                text = next(reader)
            except StopIteration:
                continue
            yield text
            pending.append(reader)
        return
    if sampling == "balanced_bytes":
        # Always read next from the source that has contributed the fewest
        # UTF-8 bytes. Unlike equal fixed quotas, this also redistributes the
        # remaining budget when a small source is exhausted.
        pending_by_size = [(0, source_id, reader) for source_id, reader in enumerate(readers)]
        heapq.heapify(pending_by_size)
        while pending_by_size:
            contributed_bytes, source_id, reader = heapq.heappop(pending_by_size)
            try:
                text = next(reader)
            except StopIteration:
                continue
            yield text
            contributed_bytes += len(text.encode("utf-8"))
            heapq.heappush(pending_by_size, (contributed_bytes, source_id, reader))
        return
    raise ValueError(
        "source_sampling must be sequential, round_robin, or balanced_bytes"
    )


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("tokenizer configuration must be a YAML mapping")
    if config.get("type") != "byte_level_bpe":
        raise ValueError("only type=byte_level_bpe is supported")
    return config


def train_command(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    trainer = BPETokenizerTrainer(
        vocab_size=args.vocab_size or int(config["vocab_size"]),
        min_frequency=int(config.get("min_frequency", 2)),
        special_tokens=config.get("special_tokens", ()),
        max_training_bytes=(
            args.max_training_bytes
            if args.max_training_bytes is not None
            else config.get("max_training_bytes")
        ),
    )

    def report(completed: int, target: int, pair: tuple[str, str], frequency: int) -> None:
        print(
            f"merges={completed}/{target} frequency={frequency} pair={pair!r}",
            file=sys.stderr,
        )

    tokenizer = trainer.train(
        iter_corpus(
            args.source or config["sources"],
            sampling=str(config.get("source_sampling", "sequential")),
        ),
        progress=report,
    )
    output_dir = args.output or Path(config.get("output_dir", "data/tokenizer-v2"))
    artifact = tokenizer.save(output_dir)
    print(
        json.dumps(
            {
                "artifact": str(artifact),
                "vocab_size": tokenizer.vocab_size,
                "merges": len(tokenizer.bpe.merges),
                "training_stats": tokenizer.metadata["training_stats"],
            },
            indent=2,
        )
    )


def inspect_command(args: argparse.Namespace) -> None:
    tokenizer = Tokenizer.load(args.tokenizer)
    identifiers = tokenizer.encode(args.text, add_bos=args.add_bos, add_eos=args.add_eos)
    print(json.dumps({"ids": identifiers, "decoded": tokenizer.decode(identifiers)}, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train and save tokenizer artifacts")
    train_parser.add_argument("--config", type=Path, default=Path("configs/tokenizer.v2.yaml"))
    train_parser.add_argument("--source", action="append", help="override source glob; repeatable")
    train_parser.add_argument("--output", type=Path)
    train_parser.add_argument("--vocab-size", type=int)
    train_parser.add_argument("--max-training-bytes", type=int)
    train_parser.set_defaults(handler=train_command)

    inspect_parser = subparsers.add_parser("inspect", help="encode and decode sample text")
    inspect_parser.add_argument("text")
    inspect_parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer-v2"))
    inspect_parser.add_argument("--add-bos", action="store_true")
    inspect_parser.add_argument("--add-eos", action="store_true")
    inspect_parser.set_defaults(handler=inspect_command)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
