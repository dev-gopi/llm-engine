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
from collections import Counter, deque
from collections.abc import Iterable, Iterator
from typing import Any

import pyarrow.parquet as pq
import regex
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


_EXTENSION_WORD = regex.compile(r"\p{L}[\p{L}\p{M}\p{N}_'’\-]{2,}")
_GRAPHEME = regex.compile(r"\X")
_PICTOGRAPH = regex.compile(r"\p{Extended_Pictographic}")


def discover_extension_tokens(
    tokenizer: Tokenizer,
    texts: Iterable[str],
    *,
    max_new_tokens: int,
    min_frequency: int,
    min_existing_tokens: int = 3,
    max_scan_bytes: int | None = None,
) -> list[str]:
    """Select frequent words/emoji that are expensive under the base tokenizer."""
    if max_new_tokens < 1 or min_frequency < 1 or min_existing_tokens < 2:
        raise ValueError("extension discovery limits must be positive")
    frequencies: Counter[str] = Counter()
    scanned_bytes = 0
    for text in texts:
        encoded_size = len(text.encode("utf-8"))
        if max_scan_bytes is not None and scanned_bytes + encoded_size > max_scan_bytes:
            break
        scanned_bytes += encoded_size
        for match in _EXTENSION_WORD.finditer(text):
            word = match.group(0)
            if match.start() > 0 and text[match.start() - 1] == " ":
                word = " " + word
            frequencies[word] += 1
        for match in _GRAPHEME.finditer(text):
            grapheme = match.group(0)
            if _PICTOGRAPH.search(grapheme):
                frequencies[grapheme] += 1

    ranked: list[tuple[int, int, str]] = []
    for token, frequency in frequencies.items():
        if frequency < min_frequency or token in tokenizer.added_tokens:
            continue
        existing_length = len(tokenizer.encode(token))
        if existing_length < min_existing_tokens:
            continue
        ranked.append((frequency * (existing_length - 1), frequency, token))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [token for _, _, token in ranked[:max_new_tokens]]


def extend_command(args: argparse.Namespace) -> None:
    config = load_config(args.config) if args.config else {}
    extension_config = config.get("extension", {})
    if extension_config and not isinstance(extension_config, dict):
        raise ValueError("tokenizer extension configuration must be a mapping")
    tokenizer_path = args.tokenizer or Path(
        extension_config.get("base_tokenizer", config.get("output_dir", "data/tokenizer-v2"))
    )
    tokenizer = Tokenizer.load(tokenizer_path)
    requested = list(args.token or ())
    if args.tokens_file:
        requested.extend(
            line.rstrip("\r\n")
            for line in args.tokens_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    sources = args.source or extension_config.get("sources", ())
    if sources:
        requested.extend(discover_extension_tokens(
            tokenizer,
            iter_corpus(sources, sampling=str(extension_config.get("source_sampling", "balanced_bytes"))),
            max_new_tokens=int(extension_config.get("max_new_tokens", 2000)),
            min_frequency=int(extension_config.get("min_frequency", 5)),
            min_existing_tokens=int(extension_config.get("min_existing_tokens", 3)),
            max_scan_bytes=extension_config.get("max_scan_bytes"),
        ))
    if not requested:
        raise ValueError("provide tokens directly or configure extension.sources")
    extended = tokenizer.extend(requested)
    tokenizer_dir = tokenizer_path.parent if tokenizer_path.name == "tokenizer.json" else tokenizer_path
    output = args.output or Path(extension_config.get(
        "output_dir", tokenizer_dir.with_name(f"{tokenizer_dir.name}-extended")
    ))
    artifact = extended.save(output)
    print(json.dumps({
        "artifact": str(artifact),
        "base_fingerprint": tokenizer.fingerprint,
        "fingerprint": extended.fingerprint,
        "old_vocab_size": tokenizer.vocab_size,
        "new_vocab_size": extended.vocab_size,
        "added_vocab_size": extended.vocab_size - tokenizer.vocab_size,
    }, indent=2))


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

    extend_parser = subparsers.add_parser(
        "extend", help="append tokens while preserving all existing token IDs"
    )
    extend_parser.add_argument("--config", type=Path, help="configuration containing an extension section")
    extend_parser.add_argument("--tokenizer", type=Path)
    extend_parser.add_argument("--source", action="append", help="dataset glob to scan; repeatable")
    extend_parser.add_argument("--token", action="append", help="token text; repeatable")
    extend_parser.add_argument("--tokens-file", type=Path, help="UTF-8 file with one token per line")
    extend_parser.add_argument(
        "--output", type=Path,
        help="output directory (defaults to <tokenizer>-extended; pass the input path for in-place)",
    )
    extend_parser.set_defaults(handler=extend_command)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
