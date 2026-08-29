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
import multiprocessing as mp
from dataclasses import asdict
from collections.abc import Iterable

import numpy as np

from datasets.filters import CorpusFilter
from datasets.loader import iter_records
from datasets.preprocessor import record_to_text
from tokenizer.encoder import Tokenizer


_WORKER_TOKENIZER: Tokenizer | None = None


def _initialize_tokenizer_worker(path: str) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = Tokenizer.load(path)


def _encode_document(text: str) -> list[int]:
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("tokenizer worker was not initialized")
    return _WORKER_TOKENIZER.encode(text, add_bos=True, add_eos=True, allowed_special="all")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--sequences-per-shard", type=int, default=8192)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="parallel tokenizer processes; filtering and output order remain deterministic",
    )
    parser.add_argument(
        "--dtype", choices=("auto", "uint16", "uint32"), default="auto",
        help="token storage type; auto uses uint16 when the vocabulary fits",
    )
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
    if args.sequence_length < 2 or args.sequences_per_shard < 1 or args.workers < 1:
        parser.error("sequence length must be >= 2; shard size and workers must be positive")

    if args.near_duplicate_distance < -1 or args.near_duplicate_distance > 15:
        parser.error("near duplicate distance must be -1 or between 0 and 15")
    if args.contamination_distance < -1 or args.contamination_distance > 15:
        parser.error("contamination distance must be -1 or between 0 and 15")
    if args.max_fingerprints < 1:
        parser.error("max fingerprints must be positive")
    tokenizer = Tokenizer.load(args.tokenizer)
    dtype_name = args.dtype
    if dtype_name == "auto":
        dtype_name = "uint16" if tokenizer.vocab_size <= np.iinfo(np.uint16).max + 1 else "uint32"
    dtype = np.dtype(dtype_name)
    if tokenizer.vocab_size - 1 > np.iinfo(dtype).max:
        parser.error(f"tokenizer vocabulary does not fit in {dtype_name}")
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
        np.asarray(sequences, dtype=dtype).tofile(args.output / name)
        shards.append({"file": name, "sequences": len(sequences)})
        sequences.clear()

    def filtered_texts() -> Iterable[str]:
        for path in args.inputs:
            for record in iter_records(path):
                text = corpus_filter.apply(record_to_text(record))
                if text is not None:
                    yield text

    if args.workers == 1:
        encoded_documents: Iterable[list[int]] = (
            tokenizer.encode(text, add_bos=True, add_eos=True, allowed_special="all")
            for text in filtered_texts()
        )
        pool = None
    else:
        pool = mp.Pool(
            args.workers,
            initializer=_initialize_tokenizer_worker,
            initargs=(str(args.tokenizer),),
        )
        encoded_documents = pool.imap(_encode_document, filtered_texts(), chunksize=32)

    try:
        for identifiers in encoded_documents:
            buffer.extend(identifiers)
            while len(buffer) >= args.sequence_length:
                sequences.append(buffer[:args.sequence_length])
                del buffer[:args.sequence_length]
                if len(sequences) >= args.sequences_per_shard:
                    flush()
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    flush()
    manifest = {
        "format": "gopi-token-shards-v1",
        "dtype": dtype_name,
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
