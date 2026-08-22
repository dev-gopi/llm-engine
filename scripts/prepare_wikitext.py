"""Convert WikiText-103 Raw Parquet shards into article-level JSONL files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

import pyarrow.parquet as pq

from datasets.preprocessor import clean

ARTICLE_TITLE = re.compile(r"^\s*=\s+[^=].*?\s+=\s*$")
def normalize_line(value: str) -> str:
    text = clean(value)
    text = re.sub(r"\s*@-@\s*", "-", text)
    text = re.sub(r"\s+@,@(?=\s|$)", ",", text)
    text = re.sub(r"\s+@\.@(?=\s|$)", ".", text)
    return text


def iter_articles(paths: list[Path]) -> Iterator[str]:
    current: list[str] = []
    for path in sorted(paths):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=["text"], batch_size=8_192):
            for row in batch.to_pylist():
                line = normalize_line(row.get("text") or "")
                if not line:
                    continue
                if ARTICLE_TITLE.fullmatch(line) and current:
                    yield "\n".join(current)
                    current = []
                current.append(line)
    if current:
        yield "\n".join(current)


def process_split(raw_dir: Path, output_dir: Path, split: str, min_characters: int) -> dict[str, int]:
    sources = sorted(raw_dir.glob(f"{split}-*.parquet"))
    if not sources:
        raise FileNotFoundError(f"no {split} Parquet shards found under {raw_dir}")
    destination = output_dir / f"{split}.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    documents = 0
    characters = 0
    with destination.open("w", encoding="utf-8") as stream:
        for article in iter_articles(sources):
            if len(article) < min_characters:
                continue
            stream.write(json.dumps({
                "id": f"wikitext-103-{split}-{documents}",
                "text": article,
                "source": "Salesforce/wikitext:wikitext-103-raw-v1",
            }, ensure_ascii=False) + "\n")
            documents += 1
            characters += len(article)
    return {"documents": documents, "characters": characters, "bytes": destination.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/wikitext-103-raw-v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/wikitext_103"))
    parser.add_argument("--min-characters", type=int, default=100)
    args = parser.parse_args()
    if args.min_characters < 1:
        parser.error("--min-characters must be positive")
    summary = {
        split: process_split(args.raw_dir, args.output_dir, split, args.min_characters)
        for split in ("train", "validation", "test")
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
