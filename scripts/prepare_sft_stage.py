"""Create a separate, complete-example SFT/replay corpus and matching training config.

Preserves original records and source files. Removes malformed/unsupported chat,
overlength chat, empty supervision, normalized exact duplicates, and held-out
prompt overlap. Raw replay text is retained (the loader can truncate it).
No semantic correctness or near-duplicate certification is implied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
import unicodedata
from collections import Counter
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

import yaml

from datasets.loader import TextDataset, iter_records
from datasets.preprocessor import record_to_text
from tokenizer.encoder import Tokenizer
from training.data import _mixture_name
from utils.config import load_yaml


def has_valid_thinking_trace(messages) -> bool:
    """Validate optional CoT traces without treating tags alone as reasoning."""
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = message["content"]
        opens = content.count("<thinking>")
        closes = content.count("</thinking>")
        if not opens and not closes:
            continue
        if opens != 1 or closes != 1:
            return False
        start = content.index("<thinking>") + len("<thinking>")
        end = content.index("</thinking>")
        if start >= end or not content[end + len("</thinking>"):].strip():
            return False
    return True


def record_key(record):
    messages = record.get("messages")
    if isinstance(messages, list):
        text = "\n".join(m["content"] for m in messages if isinstance(m, dict)
                         and m.get("role") == "user" and isinstance(m.get("content"), str))
    elif isinstance(record.get("prompt"), str):
        text = record["prompt"]
    else:
        text = record_to_text(record)
    if not text.strip():
        raise ValueError("no prompt or text")
    text = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(text.encode()).hexdigest()


def rejection_reason(record, tokenizer, max_length):
    messages = record.get("messages")
    if isinstance(record.get("prompt"), str) and isinstance(record.get("chosen"), str):
        messages = [{"role": "user", "content": record["prompt"]},
                    {"role": "assistant", "content": record["chosen"]}]
    if messages is not None:
        if not isinstance(messages, list) or not messages or any(
            not isinstance(m, dict) or m.get("role") not in {"system", "user", "assistant"}
            or not isinstance(m.get("content"), str) or not m["content"].strip() for m in messages
        ):
            return "invalid_chat"
        if messages[-1]["role"] != "assistant" or not any(m["role"] == "user" for m in messages):
            return "incomplete_chat"
        if not has_valid_thinking_trace(messages):
            return "invalid_thinking_trace"
        ids, mask = TextDataset._encode_chat(messages, tokenizer, True, True)
        if len(ids) > max_length:
            return "overlength_chat"
        if not any(mask[1:]):
            return "empty_supervision"
    else:
        # Raw replay remains a causal objective. Do not flatten chat into text,
        # which would destroy assistant-only masks and train on user prompts.
        if record.get("prepacked"):
            dataset = TextDataset([record], tokenizer, max_length=max_length)
            if not len(dataset) or not dataset[0]["loss_mask"][1:].any():
                return "empty_supervision"
        else:
            record_to_text(record)  # Validate without tokenizing a whole raw document.
    return None


def prepare(config_path: Path, output: Path, *, tokenizer_path=None, cases=()):
    config = load_yaml(config_path)
    tokenizer_path = Path(tokenizer_path or config["runtime"]["tokenizer"])
    paths = list(dict.fromkeys([*config["train_files"], *config["validation_files"]]))
    missing = [str(path) for path in [tokenizer_path, *map(Path, paths)] if not path.exists()]
    if missing:
        raise ValueError("required source artifacts unavailable: " + ", ".join(missing))
    if output.exists():
        raise ValueError("output already exists; use a new directory to preserve prior evidence")
    tokenizer = Tokenizer.load(tokenizer_path)
    names = {path: _mixture_name(path, config.get("dataset_weights", {})) for path in paths}
    for split in ("train", "validation"):
        split_names = [names[p] for p in config[f"{split}_files"]]
        if len(set(split_names)) != len(split_names):
            raise ValueError("each source within a split needs a unique dataset name")
    output.mkdir(parents=True)
    database = sqlite3.connect(output / "dedup.sqlite")
    database.execute("CREATE TABLE heldout (key TEXT PRIMARY KEY)")
    database.execute("CREATE TABLE accepted (key TEXT, split TEXT, PRIMARY KEY (key, split))")
    remapped = {}
    summary = {"tokenizer_fingerprint": tokenizer.fingerprint, "source_config": str(config_path),
               "max_sequence_length": config["max_sequence_length"], "files": [],
               "limitations": "Normalized exact prompt/document deduplication; answers are not semantically verified."}
    try:
        heldout_paths = set(config["validation_files"])
        heldout_paths.update(str(Path(p).with_name("test.jsonl")) for p in paths
                             if Path(p).with_name("test.jsonl").is_file())
        for path in sorted(heldout_paths):
            for record in iter_records(path):
                try:
                    key = record_key(record)
                except (ValueError, TypeError):
                    continue
                database.execute("INSERT OR IGNORE INTO heldout VALUES (?)", (key,))
        for path in cases:
            for record in iter_records(path):
                database.execute("INSERT OR IGNORE INTO heldout VALUES (?)", (record_key(record),))
        database.commit()
        for split in ("validation", "train"):
            for path in config[f"{split}_files"]:
                destination = output / names[path] / f"{split}.jsonl"
                destination.parent.mkdir(exist_ok=True)
                stats = Counter()
                digest = hashlib.sha256()
                with destination.open("w", encoding="utf-8") as stream:
                    for record in iter_records(path):
                        stats["read"] += 1
                        try:
                            key = record_key(record)
                            reason = rejection_reason(record, tokenizer, config["max_sequence_length"])
                        except (ValueError, TypeError, KeyError, IndexError):
                            reason = "invalid_record"
                        if reason:
                            stats[reason] += 1
                            continue
                        if any(
                            m.get("role") == "assistant" and "<thinking>" in m.get("content", "")
                            for m in record.get("messages", [])
                            if isinstance(m, dict)
                        ):
                            stats["thinking_traces"] += 1
                        if split == "train" and database.execute("SELECT 1 FROM heldout WHERE key=?", (key,)).fetchone():
                            stats["heldout_overlap"] += 1
                            continue
                        inserted = database.execute("INSERT OR IGNORE INTO accepted VALUES (?, ?)", (key, split))
                        if not inserted.rowcount:
                            stats["duplicate"] += 1
                            continue
                        line = json.dumps(record, ensure_ascii=False) + "\n"
                        stream.write(line)
                        digest.update(line.encode())
                        stats["accepted"] += 1
                database.commit()
                summary["files"].append({"source": path, "output": str(destination),
                                          "split": split, **dict(stats), "sha256": digest.hexdigest()})
                (output / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
                print(f"{split} {names[path]}: {dict(stats)}", file=sys.stderr, flush=True)
                if not stats["accepted"]:
                    raise ValueError(f"cleaning left no records for {path}; review the audit before training")
                remapped[path] = str(destination.resolve())
                manifest = Path(path).parent / "dataset-manifest.yaml"
                if manifest.exists():
                    shutil.copyfile(manifest, destination.parent / manifest.name)
        for key in ("train_files", "validation_files"):
            config[key] = [remapped[path] for path in config[key]]
        if config.get("validation_domains"):
            config["validation_domains"] = {domain: [remapped[path] for path in files]
                                              for domain, files in config["validation_domains"].items()}
        config["runtime"]["tokenizer"] = str(tokenizer_path.resolve())
        config["prepared_data"] = {"audit": str((output / "audit.json").resolve()),
                                    "tokenizer_fingerprint": tokenizer.fingerprint}
        config["require_init_from"] = True
        config["require_prepared_data"] = True
        (output / "training.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    finally:
        database.close()
    return summary


def refresh_training_config(config_path: Path, output: Path) -> Path:
    """Refresh settings while reusing an already prepared, audited corpus."""
    audit_path = output / "audit.json"
    if not audit_path.is_file():
        raise ValueError(f"prepared-data audit is missing: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    entries = audit.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("prepared-data audit contains no file mappings")
    remapped = {
        str(item["source"]): str(Path(item["output"]).resolve())
        for item in entries
        if isinstance(item, dict) and item.get("source") and item.get("output")
    }
    config = load_yaml(config_path)
    configured_paths = [*config["train_files"], *config["validation_files"]]
    missing = [path for path in configured_paths if path not in remapped]
    if missing:
        raise ValueError(
            "prepared audit does not cover configured sources: " + ", ".join(missing)
        )
    missing_outputs = [remapped[path] for path in configured_paths if not Path(remapped[path]).is_file()]
    if missing_outputs:
        raise ValueError("prepared files are missing: " + ", ".join(missing_outputs))
    for key in ("train_files", "validation_files"):
        config[key] = [remapped[path] for path in config[key]]
    if config.get("validation_domains"):
        config["validation_domains"] = {
            domain: [remapped[path] for path in paths]
            for domain, paths in config["validation_domains"].items()
        }
    tokenizer_path = Path(config["runtime"]["tokenizer"]).resolve()
    if not tokenizer_path.exists():
        raise ValueError(f"configured tokenizer is missing: {tokenizer_path}")
    config["runtime"]["tokenizer"] = str(tokenizer_path)
    config["prepared_data"] = {
        "audit": str(audit_path.resolve()),
        "tokenizer_fingerprint": audit["tokenizer_fingerprint"],
    }
    config["require_init_from"] = True
    config["require_prepared_data"] = True
    destination = output / "training.yaml"
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--cases", type=Path, action="append", default=[])
    parser.add_argument(
        "--refresh-config", action="store_true",
        help="reuse audited prepared files and update only training.yaml from the source config",
    )
    args = parser.parse_args()
    if args.refresh_config:
        if args.tokenizer or args.cases:
            parser.error("--refresh-config cannot be combined with --tokenizer or --cases")
        destination = refresh_training_config(args.training_config, args.output)
        print(f"Refreshed training config: {destination}")
        return
    prepare(args.training_config, args.output, tokenizer_path=args.tokenizer, cases=args.cases)
    print(f"Prepared training config: {args.output / 'training.yaml'}")


if __name__ == "__main__":
    main()
