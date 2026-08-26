"""Filter, deduplicate, pack, and write a corpus as bounded binary token shards."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

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
    args = parser.parse_args()
    if args.sequence_length < 2 or args.sequences_per_shard < 1:
        parser.error("sequence length must be >= 2 and sequences per shard must be positive")

    tokenizer = Tokenizer.load(args.tokenizer)
    corpus_filter = CorpusFilter(english_only=args.english_only, redact_pii=not args.keep_pii)
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
        "shards": shards,
        "filter_stats": asdict(corpus_filter.stats),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
