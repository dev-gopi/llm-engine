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
import unicodedata
from typing import Any

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

from tokenizer.encoder import Tokenizer
from utils.config import load_yaml
from datasets.loader import TextDataset
from datasets.preprocessor import record_to_text


def record_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(filter(None, (record_text(item) for item in value)))
    if isinstance(value, dict):
        preferred = ("messages", "prompt", "chosen", "response", "instruction", "input", "output", "text", "content")
        parts = [record_text(value[key]) for key in preferred if key in value]
        return "\n".join(filter(None, parts))
    return ""


def prompt_key(text: str) -> str:
    """Whitespace/case/Unicode-normalized exact prompt identity (not semantic similarity)."""
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def analyze_record(value: dict, tokenizer: Tokenizer, max_length: int) -> dict:
    """Measure the actual loader template, BOS/EOS, truncation and target mask."""
    if not isinstance(value, dict):
        raise ValueError("record must be an object")
    messages = value.get("messages")
    if isinstance(value.get("prompt"), str) and isinstance(value.get("chosen"), str):
        messages = [{"role": "user", "content": value["prompt"]},
                    {"role": "assistant", "content": value["chosen"]}]
    if isinstance(messages, list):
        ids, mask = TextDataset._encode_chat(messages, tokenizer, True, True)
        prompts = [m["content"] for m in messages if isinstance(m, dict)
                   and m.get("role") == "user" and isinstance(m.get("content"), str)]
        assistants = [m["content"] for m in messages if isinstance(m, dict)
                      and m.get("role") == "assistant" and isinstance(m.get("content"), str)]
        invalid_chat = any(not isinstance(m, dict) or m.get("role") not in {"system", "user", "assistant"}
                           or not isinstance(m.get("content"), str) or not m["content"].strip() for m in messages)
        return {"tokens": len(ids), "supervised_tokens": sum(mask[1:max_length]),
                "truncated": len(ids) > max_length, "chat": True,
                "invalid_chat": invalid_chat or not assistants or messages[-1].get("role") != "assistant",
                "missing_system": not any(isinstance(m, dict) and m.get("role") == "system" for m in messages),
                "prompts": prompts}
    if value.get("prepacked"):
        dataset = TextDataset([value], tokenizer, max_length=max_length)
        example = dataset[0]
        return {"tokens": len(example["input_ids"]), "supervised_tokens": int(example["loss_mask"][1:].sum()),
                "truncated": False, "chat": False, "invalid_chat": False, "missing_system": False, "prompts": []}
    ids = tokenizer.encode(record_to_text(value), add_bos=True, add_eos=True, allowed_special="all")
    return {"tokens": len(ids), "supervised_tokens": min(len(ids), max_length) - 1,
            "truncated": len(ids) > max_length, "chat": False,
            "invalid_chat": False, "missing_system": False, "prompts": []}


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
    parser.add_argument("--tokenizer", type=Path, help="defaults to training config runtime tokenizer")
    parser.add_argument("--cases", type=Path, action="append", default=[], help="check normalized exact benchmark prompt overlap")
    parser.add_argument("--max-records-per-file", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=Path("reports/data_quality.json"))
    args = parser.parse_args()
    paths = list(args.paths)
    max_length = 512
    train_paths: set[str] = set()
    validation_paths: set[str] = set()
    config = {}
    if args.training_config:
        config = load_yaml(args.training_config)
        paths.extend(map(Path, config.get("train_files", [])))
        paths.extend(map(Path, config.get("validation_files", [])))
        max_length = int(config.get("max_sequence_length", max_length))
        train_paths = {str(Path(p)) for p in config.get("train_files", [])}
        validation_paths = {str(Path(p)) for p in config.get("validation_files", [])}
    if not paths:
        parser.error("provide dataset paths or --training-config")
    if args.max_records_per_file < 1:
        parser.error("--max-records-per-file must be positive")
    tokenizer_path = args.tokenizer or Path(config.get("runtime", {}).get("tokenizer", "data/tokenizer"))
    if not tokenizer_path.exists():
        parser.error(f"tokenizer unavailable: {tokenizer_path}; mount the completed run's drive (do not substitute another tokenizer)")
    tokenizer = Tokenizer.load(tokenizer_path)
    benchmark_keys = set()
    for path in args.cases:
        benchmark_keys.update(prompt_key(json.loads(line)["prompt"]) for line in path.read_text().splitlines() if line.strip())
    seen: set[str] = set()
    split_records = {"train": set(), "validation": set()}
    split_prompts = {"train": set(), "validation": set()}
    per_file = []
    lengths: list[int] = []
    scripts: Counter[str] = Counter()
    duplicates = invalid = unusable = empty = truncated = sampled = 0
    missing: list[str] = []
    for path in dict.fromkeys(paths):
        if not path.is_file():
            missing.append(str(path))
            continue
        split = "train" if str(path) in train_paths else "validation" if str(path) in validation_paths else "unspecified"
        stats = Counter()
        file_lengths = []
        file_seen = set()
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream):
                if line_number >= args.max_records_per_file:
                    break
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    invalid += 1
                    stats["invalid_json"] += 1
                    continue
                text = record_text(value).strip()
                if isinstance(value, dict) and value.get("prepacked") and "token_ids" in value:
                    text = json.dumps([value.get("tokenizer_fingerprint"), value["token_ids"]])
                if not text:
                    empty += 1
                    stats["empty"] += 1
                    continue
                digest = prompt_key(text)
                duplicates += digest in seen
                seen.add(digest)
                stats["duplicates_within_file"] += digest in file_seen
                file_seen.add(digest)
                try:
                    measured = analyze_record(value, tokenizer, max_length)
                except (TypeError, ValueError, KeyError, IndexError):
                    stats["unusable_records"] += 1
                    unusable += 1
                    continue
                token_count = measured["tokens"]
                lengths.append(token_count)
                file_lengths.append(token_count)
                truncated += measured["truncated"]
                scripts[script_label(text)] += 1
                sampled += 1
                stats["sampled"] += 1
                stats["supervised_tokens"] += measured["supervised_tokens"]
                stats["zero_target_records"] += measured["supervised_tokens"] == 0
                stats["truncated"] += measured["truncated"]
                stats["chat_records"] += measured["chat"]
                stats["invalid_chat"] += measured["invalid_chat"]
                stats["chat_without_system"] += measured["missing_system"]
                stats[f"script_{script_label(text)}"] += 1
                keys = {prompt_key(prompt) for prompt in measured["prompts"]}
                stats["benchmark_prompt_matches"] += len(keys & benchmark_keys)
                if split in split_records:
                    split_records[split].add(digest)
                    split_prompts[split].update(keys)
        per_file.append({"path": str(path), "split": split, "bytes": path.stat().st_size,
                         "scan_limit": args.max_records_per_file, **dict(stats),
                         "token_length_p95": percentile(file_lengths, 0.95)})
        print(f"Audited {path}: {stats['sampled']} records", file=sys.stderr, flush=True)
    overlap = len(split_records["train"] & split_records["validation"])
    prompt_overlap = len(split_prompts["train"] & split_prompts["validation"])
    result = {
        "status": "warning" if missing or invalid or unusable or empty or duplicates or truncated or overlap or prompt_overlap
                  or any(row.get("invalid_chat", 0) or row.get("zero_target_records", 0)
                         or row.get("benchmark_prompt_matches", 0) for row in per_file) else "passed",
        "sampling": "First N lines per file, bounded diagnostic; not a complete or random corpus audit. No semantic near-duplicate detection.",
        "tokenizer": str(tokenizer_path), "tokenizer_fingerprint": tokenizer.fingerprint,
        "train_validation_record_overlap": overlap,
        "train_validation_prompt_overlap": prompt_overlap,
        "files": per_file,
        "sampled_records": sampled,
        "files_requested": len(dict.fromkeys(paths)),
        "missing_files": missing,
        "duplicates": duplicates,
        "duplicate_rate": duplicates / sampled if sampled else None,
        "invalid_json_lines": invalid,
        "unusable_records": unusable,
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
