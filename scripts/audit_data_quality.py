"""Audit JSONL training data for duplicates, lengths, truncation, and script mix."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

from tokenizer.encoder import Tokenizer
from utils.config import load_yaml


def record_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(filter(None, (record_text(item) for item in value)))
    if isinstance(value, dict):
        preferred = ("messages", "prompt", "response", "instruction", "input", "output", "text", "content")
        parts = [record_text(value[key]) for key in preferred if key in value]
        return "\n".join(filter(None, parts))
    return ""


def script_label(text: str) -> str:
    counts = Counter()
    for char in text:
        codepoint = ord(char)
        if "a" <= char.lower() <= "z":
            counts["latin"] += 1
        elif 0x0980 <= codepoint <= 0x09FF:
            counts["bengali"] += 1
        elif 0x0900 <= codepoint <= 0x097F:
            counts["devanagari"] += 1
    if not counts:
        return "other"
    label, count = counts.most_common(1)[0]
    return label if count / sum(counts.values()) >= 0.6 else "mixed"


def percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer"))
    parser.add_argument("--max-records-per-file", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=Path("reports/data_quality.json"))
    args = parser.parse_args()
    paths = list(args.paths)
    max_length = 512
    if args.training_config:
        config = load_yaml(args.training_config)
        paths.extend(map(Path, config.get("train_files", [])))
        paths.extend(map(Path, config.get("validation_files", [])))
        max_length = int(config.get("max_sequence_length", max_length))
    if not paths:
        parser.error("provide dataset paths or --training-config")
    if args.max_records_per_file < 1:
        parser.error("--max-records-per-file must be positive")
    tokenizer = Tokenizer.load(args.tokenizer)
    seen: set[str] = set()
    lengths: list[int] = []
    scripts: Counter[str] = Counter()
    duplicates = invalid = empty = truncated = sampled = 0
    missing: list[str] = []
    for path in dict.fromkeys(paths):
        if not path.is_file():
            missing.append(str(path))
            continue
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream):
                if line_number >= args.max_records_per_file:
                    break
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    invalid += 1
                    continue
                text = record_text(value).strip()
                if not text:
                    empty += 1
                    continue
                digest = hashlib.sha256(text.encode()).hexdigest()
                duplicates += digest in seen
                seen.add(digest)
                token_count = len(tokenizer.encode(text))
                lengths.append(token_count)
                truncated += token_count > max_length
                scripts[script_label(text)] += 1
                sampled += 1
    result = {
        "status": "warning" if missing or invalid or empty or duplicates or truncated else "passed",
        "sampled_records": sampled,
        "files_requested": len(dict.fromkeys(paths)),
        "missing_files": missing,
        "duplicates": duplicates,
        "duplicate_rate": duplicates / sampled if sampled else None,
        "invalid_json_lines": invalid,
        "empty_records": empty,
        "max_sequence_length": max_length,
        "records_over_max_length": truncated,
        "truncation_rate": truncated / sampled if sampled else None,
        "token_lengths": {
            "minimum": min(lengths) if lengths else None,
            "median": percentile(lengths, 0.5),
            "p95": percentile(lengths, 0.95),
            "maximum": max(lengths) if lengths else None,
        },
        "script_distribution": dict(scripts),
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
