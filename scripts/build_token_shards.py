"""Filter, deduplicate, pack, and write a corpus as bounded binary token shards."""

from __future__ import annotations

import sys
from pathlib import Path

# Remove the scripts directory before importing modules such as dataclasses.
# Python's inspect -> linecache import chain imports the standard-library
# ``tokenize`` module, which would otherwise be shadowed by scripts/tokenize.py.
script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

import argparse
import json
from dataclasses import asdict

import numpy as np

from datasets.filters import CorpusFilter
from datasets.loader import iter_records
from datasets.preprocessor import record_to_text
from tokenizer.encoder import Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--sequences-per-shard", type=int, default=8192)
    parser.add_argument("--english-only", action="store_true")
    parser.add_argument("--keep-pii", action="store_true")
    parser.add_argument(
        "--exclude", action="append", default=[], type=Path,
        help="JSON/JSONL benchmark or test file to exclude; may be repeated",
    )
    parser.add_argument(
        "--near-duplicate-distance", type=int, default=3,
        help="maximum 64-bit SimHash Hamming distance; use -1 to disable",
    )
    parser.add_argument(
        "--contamination-distance", type=int, default=8,
        help="SimHash distance for excluded benchmark text; use -1 to disable fuzzy matching",
    )
    parser.add_argument("--max-fingerprints", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.sequence_length < 2 or args.sequences_per_shard < 1:
        parser.error("sequence length must be >= 2 and sequences per shard must be positive")

    if args.near_duplicate_distance < -1 or args.near_duplicate_distance > 15:
        parser.error("near duplicate distance must be -1 or between 0 and 15")
    if args.contamination_distance < -1 or args.contamination_distance > 15:
        parser.error("contamination distance must be -1 or between 0 and 15")
    if args.max_fingerprints < 1:
        parser.error("max fingerprints must be positive")
    tokenizer = Tokenizer.load(args.tokenizer)
    excluded_texts = [
        text
        for path in args.exclude
        for record in iter_records(path)
        for text in _exclusion_texts(record)
    ]
    distance = None if args.near_duplicate_distance == -1 else args.near_duplicate_distance
    contamination_distance = (
        None if args.contamination_distance == -1 else args.contamination_distance
    )
    corpus_filter = CorpusFilter(
        english_only=args.english_only,
        redact_pii=not args.keep_pii,
        near_duplicate_distance=distance,
        excluded_texts=excluded_texts,
        contamination_distance=contamination_distance,
        max_fingerprints=args.max_fingerprints,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    buffer: list[int] = []
    sequences: list[list[int]] = []
    shards: list[dict[str, int | str]] = []

    def flush() -> None:
        if not sequences:
            return
        name = f"tokens-{len(shards):05d}.bin"
        np.asarray(sequences, dtype=np.uint32).tofile(args.output / name)
        shards.append({"file": name, "sequences": len(sequences)})
        sequences.clear()

    for path in args.inputs:
        for record in iter_records(path):
            text = corpus_filter.apply(record_to_text(record))
            if text is None:
                continue
            buffer.extend(tokenizer.encode(text, add_bos=True, add_eos=True, allowed_special="all"))
            while len(buffer) >= args.sequence_length:
                sequences.append(buffer[:args.sequence_length])
                del buffer[:args.sequence_length]
                if len(sequences) >= args.sequences_per_shard:
                    flush()
    flush()
    manifest = {
        "format": "gopi-token-shards-v1",
        "dtype": "uint32",
        "sequence_length": args.sequence_length,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "filter_config": {
            "english_only": args.english_only,
            "redact_pii": not args.keep_pii,
            "near_duplicate_distance": distance,
            "contamination_distance": contamination_distance,
            "max_fingerprints": args.max_fingerprints,
            "excluded_files": [str(path) for path in args.exclude],
        },
        "shards": shards,
        "filter_stats": asdict(corpus_filter.stats),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def _exclusion_texts(record: dict) -> list[str]:
    try:
        return [record_to_text(record)]
    except ValueError:
        values: list[str] = []
        for key in ("prompt", "chosen", "rejected"):
            if isinstance(record.get(key), str) and record[key].strip():
                values.append(record[key])
        expected = record.get("expected")
        if isinstance(expected, list):
            values.extend(str(value) for value in expected if str(value).strip())
        if not values:
            raise ValueError("exclusion record contains no supported benchmark text")
        return values


if __name__ == "__main__":
    main()
