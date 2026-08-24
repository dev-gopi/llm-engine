"""Download and prepare Hugging Face datasets into raw and processed LLM Engine JSONL format."""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
from pathlib import Path

# Avoid resolving the standard-library ``tokenize`` module to scripts/tokenize.py
script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

import pyarrow.parquet as pq

DEFAULT_BOT_NAME = "Gopi"


def extract_messages(row: dict, dataset_name: str, bot_name: str = DEFAULT_BOT_NAME) -> list[dict[str, str]] | None:
    system_msg = {
        "role": "system",
        "content": f"You are {bot_name}, a helpful, honest, and friendly AI assistant.",
    }

    # Case 1: Pre-formatted chat messages (e.g. UltraChat, NoRobots, OpenAssistant)
    if "messages" in row and isinstance(row["messages"], list):
        msgs = []
        for msg in row["messages"]:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                msgs.append({"role": str(msg["role"]), "content": str(msg["content"]).strip()})
        if msgs and msgs[0]["role"] != "system":
            msgs.insert(0, system_msg)
        return msgs if len(msgs) >= 2 else None

    # Case 2: Instruction / Context / Output format (e.g. Dolly 15k, Alpaca, Platypus)
    instruction = row.get("instruction") or row.get("prompt") or row.get("input_text") or row.get("question")
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


def fetch_hf_parquet_urls(dataset_name: str, config: str = "default", split: str = "train") -> list[str]:
    api_url = f"https://huggingface.co/api/datasets/{dataset_name}/parquet/{config}/{split}"
    request = urllib.request.Request(api_url, headers={"User-Agent": "LLMEngine/1.0"})
    try:
        with urllib.request.urlopen(request) as response:
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
    train_size: int = 10000,
    validation_size: int = 1000,
    test_size: int = 1000,
    bot_name: str = DEFAULT_BOT_NAME,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)

    urls = fetch_hf_parquet_urls(dataset_name, config=config, split=split)
    print(f"Fetching {dataset_name} ({len(urls)} parquet files)...")

    records: list[dict] = []
    total_needed = train_size + validation_size + test_size

    for index, url in enumerate(urls):
        if len(records) >= total_needed:
            break
        print(f"Downloading {url}...")
        request = urllib.request.Request(url, headers={"User-Agent": "LLMEngine/1.0"})
        with urllib.request.urlopen(request) as response:
            content = response.read()

            # Save raw file locally if raw_dir is specified
            if raw_dir is not None:
                raw_file_path = raw_dir / f"{split}-{index:05d}.parquet"
                raw_file_path.write_bytes(content)
                print(f"Saved raw dataset file to {raw_file_path}")

            buffer = io.BytesIO(content)
            parquet_file = pq.ParquetFile(buffer)
            for batch in parquet_file.iter_batches():
                for row in batch.to_pylist():
                    messages = extract_messages(row, dataset_name, bot_name=bot_name)
                    record = {
                        "id": f"{dataset_name.replace('/', '_')}_{len(records)}",
                        "source": dataset_name,
                    }
                    if messages:
                        record.update({"bot_name": bot_name, "messages": messages})
                    else:
                        text = row.get("text") or row.get("content") or row.get("code")
                        # Row-wise conversation exports (for example OASST1) must
                        # be reconstructed into turns instead of treated as prose.
                        if row.get("role") or not isinstance(text, str) or not text.strip():
                            continue
                        record["text"] = text.strip()
                    records.append(record)
                    if len(records) >= total_needed:
                        break

    actual_total = len(records)
    if actual_total == 0:
        raise ValueError(f"No valid records found in {dataset_name}")

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
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--validation-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--bot-name", default=DEFAULT_BOT_NAME)
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
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        bot_name=args.bot_name,
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
