"""Clean JSONL corpora and write an auditable, non-sharded JSONL output."""

from __future__ import annotations

import sys
from pathlib import Path

# Avoid shadowing the standard-library tokenize module with scripts/tokenize.py.
script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

import argparse
import hashlib
import json
import statistics
from collections import Counter
from dataclasses import asdict
from typing import Any, Iterator

from datasets.filters import CorpusFilter
from datasets.loader import iter_records
from datasets.preprocessor import record_to_text
from tokenizer.encoder import Tokenizer


def detect_language(text: str) -> str:
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return "unknown"
    bengali = sum("\u0980" <= character <= "\u09ff" for character in letters)
    devanagari = sum("\u0900" <= character <= "\u097f" for character in letters)
    ascii_letters = sum(character.isascii() for character in letters)
    code_markers = sum(
        marker in text for marker in ("def ", "class ", "import ", "from ", "function ", "#include", "const ", "let ", "var ")
    )
    if code_markers >= 2:
        return "code"
    dominant = max(bengali, devanagari, ascii_letters)
    if dominant / len(letters) < 0.70:
        return "mixed"
    if dominant == bengali:
        return "bn"
    if dominant == devanagari:
        return "hi"
    return "en"


def _excluded_texts(paths: list[Path]) -> Iterator[str]:
    for path in paths:
        for record in iter_records(path):
            try:
                yield record_to_text(record)
            except (TypeError, ValueError):
                continue


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    return sorted(values)[min(len(values) - 1, int((len(values) - 1) * fraction))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--exclude", action="append", default=[], type=Path)
    parser.add_argument("--english-only", action="store_true")
    parser.add_argument("--near-duplicate-distance", type=int, default=3)
    parser.add_argument("--max-fingerprints", type=int, default=5_000_000)
    args = parser.parse_args()

    tokenizer = Tokenizer.load(args.tokenizer)
    corpus_filter = CorpusFilter(
        english_only=args.english_only,
        redact_pii=True,
        near_duplicate_distance=args.near_duplicate_distance,
        excluded_texts=_excluded_texts(args.exclude),
        max_fingerprints=args.max_fingerprints,
        preserve_whitespace=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    lengths: list[int] = []
    languages: Counter[str] = Counter()
    digest = hashlib.sha256()
    input_records = 0
    with temporary.open("w", encoding="utf-8") as stream:
        for input_path in args.inputs:
            for record in iter_records(input_path):
                if record.get("prepacked"):
                    raise ValueError("clean original documents before packing")
                input_records += 1
                try:
                    original = record_to_text(record)
                except (TypeError, ValueError):
                    corpus_filter.stats.empty += 1
                    continue
                cleaned = corpus_filter.apply(original)
                if cleaned is None:
                    continue
                language = detect_language(cleaned)
                token_count = len(tokenizer.encode(cleaned, add_bos=True, add_eos=True))
                output: dict[str, Any] = dict(record)
                # Cleaned corpora use a single explicit text representation so
                # every accepted byte is exactly what the audit measured.
                output.pop("messages", None)
                output.pop("prompt", None)
                output.pop("chosen", None)
                output.pop("rejected", None)
                output["text"] = cleaned
                output["language"] = language
                output["token_count"] = token_count
                line = json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n"
                stream.write(line)
                digest.update(line.encode("utf-8"))
                lengths.append(token_count)
                languages[language] += 1
                if input_records % 10_000 == 0:
                    print(f"processed={input_records} accepted={len(lengths)}", file=sys.stderr, flush=True)
    temporary.replace(args.output)

    truncated = sum(length > args.max_length for length in lengths)
    audit = {
        "format": "clean-jsonl-audit-v1",
        "inputs": [str(path) for path in args.inputs],
        "output": str(args.output),
        "output_sha256": digest.hexdigest(),
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "input_records": input_records,
        "output_records": len(lengths),
        "filter_stats": asdict(corpus_filter.stats),
        "languages": dict(sorted(languages.items())),
        "token_lengths": {
            "minimum": min(lengths, default=0),
            "median": statistics.median(lengths) if lengths else 0,
            "p95": _percentile(lengths, 0.95),
            "maximum": max(lengths, default=0),
            "truncated_at": args.max_length,
            "truncated_records": truncated,
            "truncated_percent": round(100 * truncated / len(lengths), 4) if lengths else 0.0,
        },
        "excluded_splits": [str(path) for path in args.exclude],
    }
    audit_path = args.output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
