"""Download a bounded multilingual Wikipedia corpus for local RAG."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

import pyarrow.parquet as pq


DATASET = "wikimedia/wikipedia"
SNAPSHOT = "20231101"
DEFAULT_LANGUAGES = ("simple", "bn", "hi")


def parquet_urls(language: str, *, timeout: float = 30.0) -> list[str]:
    endpoint = (
        f"https://huggingface.co/api/datasets/{DATASET}/parquet/"
        f"{SNAPSHOT}.{language}/train"
    )
    request = urllib.request.Request(endpoint, headers={"User-Agent": "Gopi-RAG/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"no parquet files returned for language {language!r}")
    return [str(url) for url in payload]


def download_language(
    language: str, destination: Path, *, max_articles: int, timeout: float = 120.0
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = destination.with_suffix(destination.suffix + ".tmp")
    count = 0
    with temporary_output.open("w", encoding="utf-8") as output:
        for file_index, url in enumerate(parquet_urls(language, timeout=timeout)):
            if count >= max_articles:
                break
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f"gopi-wikipedia-{language}-{file_index:02d}-", suffix=".parquet"
            )
            os.close(descriptor)
            parquet_path = Path(temporary_name)
            try:
                print(f"Downloading {language}: {url}", flush=True)
                request = urllib.request.Request(url, headers={"User-Agent": "Gopi-RAG/0.1"})
                with urllib.request.urlopen(request, timeout=timeout) as response, parquet_path.open("wb") as stream:
                    shutil.copyfileobj(response, stream, length=1024 * 1024)
                for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=256):
                    for row in batch.to_pylist():
                        text = str(row.get("text") or "").strip()
                        if not text:
                            continue
                        output.write(json.dumps({
                            "id": str(row.get("id") or f"{language}-{count}"),
                            "language": language,
                            "title": str(row.get("title") or "Untitled"),
                            "url": str(row.get("url") or ""),
                            "text": text,
                            "source": DATASET,
                            "snapshot": SNAPSHOT,
                            "license": "CC-BY-SA-3.0 and GFDL",
                        }, ensure_ascii=False) + "\n")
                        count += 1
                        if count >= max_articles:
                            break
                    if count >= max_articles:
                        break
            finally:
                parquet_path.unlink(missing_ok=True)
    temporary_output.replace(destination)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", default=list(DEFAULT_LANGUAGES))
    parser.add_argument("--articles-per-language", type=int, default=10_000)
    parser.add_argument("--output-dir", type=Path, default=Path("data/rag/wikipedia"))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--minimum-free-gb", type=float, default=5.0)
    args = parser.parse_args()
    if args.articles_per_language < 1:
        parser.error("--articles-per-language must be positive")
    free_bytes = shutil.disk_usage(Path.cwd()).free
    if free_bytes < args.minimum_free_gb * 1024**3:
        parser.error(
            f"only {free_bytes / 1024**3:.1f} GiB is free; "
            f"at least {args.minimum_free_gb:.1f} GiB is required"
        )
    counts = {}
    for language in args.languages:
        counts[language] = download_language(
            language,
            args.output_dir / f"wikipedia-{language}.jsonl",
            max_articles=args.articles_per_language,
            timeout=args.timeout,
        )
        print(f"Prepared {language}: {counts[language]} articles", flush=True)
    manifest = {
        "dataset": DATASET,
        "snapshot": SNAPSHOT,
        "license": ["CC-BY-SA-3.0", "GFDL"],
        "languages": counts,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), **manifest}, indent=2))


if __name__ == "__main__":
    main()
