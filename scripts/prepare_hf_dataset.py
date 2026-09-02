"""Download and prepare Hugging Face datasets into raw and processed LLM Engine JSONL format."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Sequence

# Avoid resolving the standard-library ``tokenize`` module to scripts/tokenize.py
script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

import pyarrow.parquet as pq

DEFAULT_BOT_NAME = "Gopi"
PREFERENCE_SCORE_FIELDS = (
    "helpfulness", "correctness", "coherence", "complexity", "verbosity",
)


def normalize_conversation(messages: list, system_msg: dict[str, str]) -> list[dict[str, str]] | None:
    role_aliases = {"human": "user", "gpt": "assistant", "bot": "assistant"}
    normalized: list[dict[str, str]] = []
    tool_catalogs: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or message.get("from") or "").lower()
        content = message.get("content", message.get("value", ""))
        if role == "tool_catalog" and str(content).strip():
            tool_catalogs.append(str(content).strip())
            continue
        role = role_aliases.get(role, role)
        if role in {"system", "user", "assistant"} and str(content).strip():
            normalized.append({"role": role, "content": str(content).strip()})
    if tool_catalogs:
        catalog = "Available tools (JSON):\n" + "\n".join(tool_catalogs)
        if normalized and normalized[0]["role"] == "system":
            normalized[0]["content"] += "\n\n" + catalog
        else:
            normalized.insert(0, {"role": "system", "content": system_msg["content"] + "\n\n" + catalog})
    elif normalized and normalized[0]["role"] != "system":
        normalized.insert(0, system_msg)
    return normalized if len(normalized) >= 2 else None


def extract_messages(row: dict, dataset_name: str, bot_name: str = DEFAULT_BOT_NAME) -> list[dict[str, str]] | None:
    system_msg = {
        "role": "system",
        "content": f"You are {bot_name}, a helpful, honest, and friendly AI assistant.",
    }

    # Case 1: Pre-formatted chat messages (e.g. UltraChat, NoRobots, OpenAssistant)
    if "messages" in row and isinstance(row["messages"], list):
        return normalize_conversation(row["messages"], system_msg)

    # Case 2: ShareGPT-style conversations, including function/tool catalogs.
    if "conversations" in row and isinstance(row["conversations"], list):
        return normalize_conversation(row["conversations"], system_msg)

    # Case 3: Instruction / Context / Output format (e.g. Dolly 15k, Alpaca, Platypus)
    instruction = (
        row.get("instruction")
        or row.get("prompt")
        or row.get("input_text")
        or row.get("question")
        or row.get("query")
    )
    output = row.get("response") or row.get("output") or row.get("completion") or row.get("answer")
    context = row.get("context") or row.get("input", "")

    if instruction and output and str(instruction).strip() and str(output).strip():
        instruction_str = str(instruction).strip()
        output_str = str(output).strip()
        context_str = str(context).strip() if context else ""

        user_content = f"{instruction_str}\n\nContext:\n{context_str}" if context_str else instruction_str
        return [
            system_msg,
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output_str},
        ]

    # # Case 3: Plain text / Code records
    # text = row.get("text") or row.get("content") or row.get("code")
    # if text and isinstance(text, str) and text.strip():
    #     return [{"role": "system", "content": text.strip()}]

    return None


def fetch_hf_parquet_urls(
    dataset_name: str, config: str = "default", split: str = "train",
    *, timeout: float = 60.0,
) -> list[str]:
    api_url = f"https://huggingface.co/api/datasets/{dataset_name}/parquet/{config}/{split}"
    request = urllib.request.Request(api_url, headers={"User-Agent": "LLMEngine/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
            if isinstance(data, list):
                return data
    except Exception as error:
        print(f"Notice: Could not list API endpoints for {dataset_name} ({error}). Checking fallback...")
    return [f"https://huggingface.co/datasets/{dataset_name}/resolve/main/{split}.parquet"]


def download_and_convert_dataset(
    dataset_name: str,
    output_dir: Path,
    *,
    raw_dir: Path | None = None,
    config: str = "default",
    split: str = "train",
    train_size: int | None = 10000,
    validation_size: int | None = 1000,
    test_size: int | None = 1000,
    bot_name: str = DEFAULT_BOT_NAME,
    timeout: float = 60.0,
    exclude_source_patterns: Sequence[str] = (),
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)

    sizes = (train_size, validation_size, test_size)
    if any(size is None for size in sizes) and not all(size is None for size in sizes):
        raise ValueError("provide all three split sizes or omit all three for the full dataset")
    if any(size is not None and size < 0 for size in sizes):
        raise ValueError("split sizes must be non-negative")
    full_dataset = all(size is None for size in sizes)
    total_needed = None if full_dataset else sum(size for size in sizes if size is not None)
    if timeout <= 0:
        raise ValueError("download timeout must be positive")
    urls = fetch_hf_parquet_urls(
        dataset_name, config=config, split=split, timeout=timeout
    )
    print(f"Fetching {dataset_name} ({len(urls)} parquet files)...")
    records: list[dict] = []
    counts = {"train": 0, "validation": 0, "test": 0}
    streams = {
        name: (output_dir / f".{name}.jsonl.tmp").open("w", encoding="utf-8")
        for name in counts
    } if full_dataset else {}

    try:
        for index, url in enumerate(urls):
            if total_needed is not None and len(records) >= total_needed:
                break
            cached_path = raw_dir / f"{split}-{index:05d}.parquet" if raw_dir is not None else None
            if cached_path is not None and cached_path.is_file():
                print(f"Reusing raw dataset file {cached_path}")
                parquet_path = cached_path
                temporary_name = str(cached_path)
                remove_temporary = False
            else:
                print(f"Downloading {url}...")
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{split}-{index:05d}.", suffix=".parquet",
                    dir=raw_dir or output_dir,
                )
                os.close(descriptor)
                parquet_path = Path(temporary_name)
                remove_temporary = True
            try:
                if remove_temporary:
                    request = urllib.request.Request(url, headers={"User-Agent": "LLMEngine/1.0"})
                    with urllib.request.urlopen(request, timeout=timeout) as response, parquet_path.open("wb") as output:
                        shutil.copyfileobj(response, output, length=1024 * 1024)
                    if raw_dir is not None:
                        raw_file_path = raw_dir / f"{split}-{index:05d}.parquet"
                        os.replace(parquet_path, raw_file_path)
                        parquet_path = raw_file_path
                        remove_temporary = False
                        print(f"Saved raw dataset file to {raw_file_path}")

                parquet_file = pq.ParquetFile(parquet_path)
                for batch in parquet_file.iter_batches():
                    for row in batch.to_pylist():
                        upstream_source = str(row.get("source") or row.get("resource") or "")
                        if any(pattern.lower() in upstream_source.lower() for pattern in exclude_source_patterns):
                            continue
                        messages = extract_messages(row, dataset_name, bot_name=bot_name)
                        record_index = sum(counts.values()) if full_dataset else len(records)
                        record = {
                            "id": f"{dataset_name.replace('/', '_')}_{record_index}",
                            "source": dataset_name,
                        }
                        if upstream_source:
                            record["source_subset"] = upstream_source
                        if messages:
                            record.update({"bot_name": bot_name, "messages": messages})
                        else:
                            text = row.get("text") or row.get("content") or row.get("code")
                            if row.get("role") or not isinstance(text, str) or not text.strip():
                                continue
                            record["text"] = text.strip()
                        scores = {
                            field: row[field]
                            for field in PREFERENCE_SCORE_FIELDS
                            if row.get(field) is not None
                        }
                        if scores:
                            record["scores"] = scores
                        if full_dataset:
                            messages_for_split = record.get("messages")
                            if scores and isinstance(messages_for_split, list):
                                # Keep every rated candidate for one prompt in the
                                # same split so DPO pairing remains possible and
                                # prompts cannot leak across train and validation.
                                split_key = [
                                    message for message in messages_for_split
                                    if message.get("role") != "assistant"
                                ]
                            else:
                                split_key = messages_for_split or record.get("text", "")
                            fingerprint = hashlib.sha256(json.dumps(split_key, sort_keys=True, ensure_ascii=False).encode()).digest()[0]
                            split_name = "validation" if fingerprint < 13 else "test" if fingerprint < 26 else "train"
                            streams[split_name].write(json.dumps(record, ensure_ascii=False) + "\n")
                            counts[split_name] += 1
                        else:
                            records.append(record)
                            if len(records) >= total_needed:
                                break
            finally:
                if remove_temporary:
                    Path(temporary_name).unlink(missing_ok=True)
    finally:
        for stream in streams.values():
            stream.close()

    if full_dataset:
        if not sum(counts.values()):
            raise ValueError(f"No valid records found in {dataset_name}")
        for name in counts:
            (output_dir / f".{name}.jsonl.tmp").replace(output_dir / f"{name}.jsonl")
        return counts

    actual_total = len(records)
    if actual_total == 0:
        raise ValueError(f"No valid records found in {dataset_name}")

    # Source Parquet files are often ordered by topic or collection time.
    # Hash-order before slicing to avoid systematic train/validation drift.
    records.sort(key=lambda record: hashlib.sha256(
        json.dumps(
            record.get("messages", record.get("text", "")),
            sort_keys=True, ensure_ascii=False,
        ).encode("utf-8")
    ).digest())

    actual_train = min(train_size, int(actual_total * 0.8)) if actual_total < total_needed else train_size
    actual_val = min(validation_size, int(actual_total * 0.1)) if actual_total < total_needed else validation_size
    actual_test = max(0, actual_total - actual_train - actual_val)

    train_records = records[:actual_train]
    val_records = records[actual_train : actual_train + actual_val]
    test_records = records[actual_train + actual_val : actual_train + actual_val + actual_test]

    counts = {}
    for name, subset in (("train", train_records), ("validation", val_records), ("test", test_records)):
        file_path = output_dir / f"{name}.jsonl"
        with file_path.open("w", encoding="utf-8") as stream:
            for record in subset:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        counts[name] = len(subset)

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset identifier (e.g. databricks/databricks-dolly-15k)")
    parser.add_argument("--config", default="default", help="Dataset configuration / subset name")
    parser.add_argument("--split", default="train", help="Dataset split to download")
    parser.add_argument("--raw-dir", type=Path, default=None, help="Directory to save raw Parquet files")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory path for processed JSONL files")
    parser.add_argument("--train-size", type=int, default=10000, help="Bounded train rows (default: 10000)")
    parser.add_argument("--validation-size", type=int, default=1000, help="Bounded validation rows (default: 1000)")
    parser.add_argument("--test-size", type=int, default=1000, help="Bounded test rows (default: 1000)")
    parser.add_argument("--full", action="store_true", help="Process the complete source split using deterministic 90/5/5 splits")
    parser.add_argument("--bot-name", default=DEFAULT_BOT_NAME)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--exclude-source-pattern", action="append", default=[],
        help="Skip rows whose source/resource field contains this text (repeatable)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folder_name = args.dataset.split("/")[-1].lower()
    raw_dir = args.raw_dir or Path("data/raw") / folder_name
    output_dir = args.output_dir or Path("data/processed") / folder_name

    counts = download_and_convert_dataset(
        dataset_name=args.dataset,
        output_dir=output_dir,
        raw_dir=raw_dir,
        config=args.config,
        split=args.split,
        train_size=None if args.full else args.train_size,
        validation_size=None if args.full else args.validation_size,
        test_size=None if args.full else args.test_size,
        bot_name=args.bot_name,
        timeout=args.timeout,
        exclude_source_patterns=args.exclude_source_pattern,
    )
    result = {
        "dataset": args.dataset,
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "records": counts,
        "train_file": str(output_dir / "train.jsonl"),
        "validation_file": str(output_dir / "validation.jsonl"),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
