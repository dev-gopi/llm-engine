"""Pack cleaned JSONL documents into one non-sharded JSONL corpus."""

from __future__ import annotations

import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

import argparse
import json
from typing import Any

from datasets.loader import iter_records
from datasets.preprocessor import record_to_text
from tokenizer.encoder import Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--sequence-length", type=int, default=512)
    args = parser.parse_args()
    if args.sequence_length < 2:
        parser.error("--sequence-length must be at least 2")

    tokenizer = Tokenizer.load(args.tokenizer)
    eos = tokenizer.token_to_id("<|eos|>")
    if eos is None:
        raise ValueError("tokenizer does not define <|eos|>")
    capacity = args.sequence_length - 1  # The loader adds BOS at training time.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    buffer: list[int] = []
    sources: set[str] = set()
    document_count = 0
    output_records = 0
    split_chunks = 0

    def write_record(stream, identifiers: list[int], docs: int, source_names: set[str]) -> None:
        nonlocal output_records
        output_records += 1
        row: dict[str, Any] = {
            "id": f"packed-{output_records:09d}",
            "source": sorted(source_names),
            "text": tokenizer.decode(identifiers),
            "prepacked": True,
            "document_count": docs,
            "token_count": len(identifiers) + 1,
        }
        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    with temporary.open("w", encoding="utf-8") as stream:
        for input_path in args.inputs:
            for record in iter_records(input_path):
                document = tokenizer.encode(record_to_text(record), allowed_special="all")
                source = str(record.get("source", input_path.stem))
                pieces = [document[index:index + capacity - 1] + [eos]
                          for index in range(0, len(document), capacity - 1)] or [[eos]]
                split_chunks += max(0, len(pieces) - 1)
                for piece in pieces:
                    if buffer and len(buffer) + len(piece) > capacity:
                        write_record(stream, buffer, document_count, sources)
                        buffer, sources, document_count = [], set(), 0
                    buffer.extend(piece)
                    sources.add(source)
                    document_count += 1
                    if len(buffer) == capacity:
                        write_record(stream, buffer, document_count, sources)
                        buffer, sources, document_count = [], set(), 0
        if buffer:
            write_record(stream, buffer, document_count, sources)
    temporary.replace(args.output)
    metadata = {
        "format": "packed-jsonl-v1",
        "inputs": [str(path) for path in args.inputs],
        "output": str(args.output),
        "sequence_length": args.sequence_length,
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "vocab_size": tokenizer.vocab_size,
        "output_records": output_records,
        "split_chunks": split_chunks,
        "boundary_token": "<|eos|>",
        "loader_behavior": "BOS is added; EOS is already embedded",
    }
    args.output.with_suffix(".packing.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
