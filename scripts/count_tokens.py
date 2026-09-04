import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[0] = str(ROOT)

from src.tokenizer import Tokenizer

TOKENIZER_PATH = ROOT / "data/tokenizer/tokenizer.json"

DATASETS = {
    "tinystories": ROOT / "data/processed/tinystories/train.jsonl",
    "wikitext_103": ROOT / "data/processed/wikitext_103/train.jsonl",
}

tokenizer = Tokenizer.load(TOKENIZER_PATH)

def get_text(record):
    # Adjust this if your JSONL uses another field name
    for key in ["text", "content", "input"]:
        if key in record and record[key] is not None:
            return record[key]

    raise ValueError(f"Cannot find text field. Keys: {record.keys()}")


def count_tokens(file_path: Path, batch_size: int = 1000) -> tuple[int, int]:
    total_tokens = 0
    total_samples = 0
    batch = []

    print(f"\nCounting: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)
            text = get_text(record)

            batch.append(text)
            total_samples += 1

            if len(batch) >= batch_size:
                total_tokens += sum(len(tokenizer.encode(text)) for text in batch)

                batch = []

    # Process remaining texts
    if batch:
        total_tokens += sum(len(tokenizer.encode(text)) for text in batch)

    return total_samples, total_tokens


results = {}

for name, file_path in DATASETS.items():
    samples, tokens = count_tokens(file_path)

    results[name] = {
        "samples": samples,
        "tokens": tokens,
    }

    print(f"\n{name.upper()}")
    print(f"Samples: {samples:,}")
    print(f"Tokens:  {tokens:,} ({tokens / 1e9:.4f}B)")


print("\n" + "=" * 60)
print("TOTAL")

total_samples = sum(x["samples"] for x in results.values())
total_tokens = sum(x["tokens"] for x in results.values())

print(f"Samples: {total_samples:,}")
print(f"Tokens:  {total_tokens:,} ({total_tokens / 1e9:.4f}B)")
print("=" * 60)