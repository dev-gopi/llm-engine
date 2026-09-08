"""Prepare Hugging Face or raw-web text into filtered, sharded LLM JSONL.

The YAML schema mirrors a DataTrove pipeline: readers -> extraction -> filters
-> language/PII/deduplication -> statistics -> writers.  It deliberately keeps
the final JSONL format portable, so the result can be passed directly to
``scripts/tokenize.py`` or ``scripts/build_token_shards.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

import argparse
import hashlib
import html
import json
import re
from collections import Counter
from dataclasses import asdict
from html.parser import HTMLParser
from typing import Any, Iterable, Iterator

import importlib.metadata
import importlib.util

from datasets.filters import CorpusFilter
from utils.config import load_yaml


class _HTMLText(HTMLParser):
    """Small dependency-free HTML extractor for JSONL web-crawl records."""

    ignored = {"script", "style", "noscript", "svg", "template"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.ignored:
            self._ignored += 1
        elif tag in {"p", "br", "div", "li", "article", "section", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.ignored and self._ignored:
            self._ignored -= 1
        elif tag in {"p", "div", "li", "article", "section"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored:
            self.parts.append(data)


def extract_html(value: str) -> str:
    parser = _HTMLText()
    parser.feed(value)
    parser.close()
    return html.unescape(" ".join("".join(parser.parts).split()))


def detect_language(text: str) -> str:
    """Conservative built-in detector; use DataTrove's fastText filter at scale."""
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return "unknown"
    ascii_ratio = sum(char.isascii() for char in letters) / len(letters)
    bengali = sum("\u0980" <= char <= "\u09ff" for char in letters) / len(letters)
    devanagari = sum("\u0900" <= char <= "\u097f" for char in letters) / len(letters)
    if ascii_ratio >= .85:
        return "en"
    if bengali >= .70:
        return "bn"
    if devanagari >= .70:
        return "hi"
    return "other"


def _field(record: dict[str, Any], fields: Iterable[str]) -> str | None:
    for field in fields:
        value: Any = record
        for part in field.split("."):
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(part)
        if isinstance(value, str) and value.strip():
            return value
    return None


def local_records(paths: list[str]) -> Iterator[dict[str, Any]]:
    for raw_path in paths:
        path = Path(raw_path)
        if path.suffix.lower() not in {".jsonl", ".json"}:
            raise ValueError("web reader supports JSONL/JSON; use DataTrove's WARC reader for WARC files")
        with path.open(encoding="utf-8") as handle:
            if path.suffix.lower() == ".json":
                values = json.load(handle)
                if not isinstance(values, list):
                    raise ValueError(f"JSON source must be a list: {path}")
                for value in values:
                    if isinstance(value, dict):
                        yield value
            else:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"invalid JSON at {path}:{line_number}") from error
                    if isinstance(value, dict):
                        yield value


def hf_records(source: dict[str, Any]) -> Iterator[dict[str, Any]]:
    # This project has an internal ``datasets`` package.  The Hugging Face
    # package intentionally has the same import name, so resolve its installed
    # distribution explicitly when the internal package is first on sys.path.
    try:
        import datasets as datasets_module
        load_dataset = datasets_module.load_dataset
    except AttributeError:
        try:
            package_root = Path(importlib.metadata.distribution("datasets").locate_file("datasets"))
            spec = importlib.util.spec_from_file_location(
                "datasets", package_root / "__init__.py",
                submodule_search_locations=[str(package_root)],
            )
            if spec is None or spec.loader is None:
                raise ImportError("could not resolve Hugging Face datasets")
            # CorpusFilter has already been imported above, so replacing this
            # module entry cannot affect the preparation filters.
            for name in [name for name in sys.modules if name == "datasets" or name.startswith("datasets.")]:
                del sys.modules[name]
            datasets_module = importlib.util.module_from_spec(spec)
            sys.modules["datasets"] = datasets_module
            spec.loader.exec_module(datasets_module)
            load_dataset = datasets_module.load_dataset
        except (ImportError, importlib.metadata.PackageNotFoundError) as error:
            raise RuntimeError("Hugging Face input requires `pip install -e '.[data]'`") from error
    except ImportError as error:
        raise RuntimeError("Hugging Face input requires `pip install -e '.[data]'`") from error
    name = source.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("source.name is required for source.kind: huggingface")
    kwargs: dict[str, Any] = {"split": source.get("split", "train")}
    if source.get("config") is not None:
        kwargs["name"] = source["config"]
    if source.get("revision") is not None:
        kwargs["revision"] = source["revision"]
    if source.get("streaming", True):
        kwargs["streaming"] = True
    dataset = load_dataset(name, **kwargs)
    limit = source.get("limit")
    for index, row in enumerate(dataset):
        if limit is not None and index >= int(limit):
            break
        yield dict(row)


def build_filter(config: dict[str, Any]) -> CorpusFilter:
    quality = config.get("quality", {})
    dedup = config.get("deduplication", {})
    pii = config.get("pii", {})
    if not all(isinstance(item, dict) for item in (quality, dedup, pii)):
        raise ValueError("quality, deduplication, and pii must be mappings")
    return CorpusFilter(
        min_chars=int(quality.get("min_chars", 40)),
        max_chars=int(quality.get("max_chars", 100_000)),
        redact_pii=bool(pii.get("redact", True)),
        near_duplicate_distance=(None if dedup.get("near_duplicate_distance", 3) is None
                                 else int(dedup.get("near_duplicate_distance", 3))),
        max_fingerprints=int(dedup.get("max_fingerprints", 1_000_000)),
    )


def run_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    source = config.get("source", {})
    output = config.get("output", {})
    if not isinstance(source, dict) or not isinstance(output, dict):
        raise ValueError("source and output must be mappings")
    kind = source.get("kind", "huggingface")
    if kind == "web" and (not isinstance(source.get("paths"), list) or not source["paths"]):
        raise ValueError("source.paths must be a non-empty list for source.kind: web")
    records = hf_records(source) if kind == "huggingface" else local_records(list(source.get("paths", []))) if kind == "web" else None
    if records is None:
        raise ValueError("source.kind must be 'huggingface' or 'web'")
    destination = Path(output.get("directory", "data/processed/corpus"))
    shard_size = int(output.get("shard_size", 100_000))
    if shard_size < 1:
        raise ValueError("output.shard_size must be positive")
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.glob("shard-*.jsonl")):
        raise FileExistsError(
            f"output already has shards: {destination}; choose a new output.directory"
        )
    fields = source.get("text_fields", ["text", "content", "raw_content", "html"])
    if not isinstance(fields, list) or not all(isinstance(item, str) for item in fields):
        raise ValueError("source.text_fields must be a list of field names")
    extraction = config.get("extraction", {})
    languages = set(config.get("languages", {}).get("allow", []))
    blocked = [re.compile(pattern, re.IGNORECASE) for pattern in config.get("unwanted_content", {}).get("patterns", [])]
    corpus_filter = build_filter(config)
    counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    output_files: list[str] = []
    stream = None
    try:
        for index, raw in enumerate(records):
            counts["input"] += 1
            text = _field(raw, fields)
            if text is None:
                counts["missing_text"] += 1
                continue
            if extraction.get("html", False) or "html" in fields and "<" in text:
                text = extract_html(text)
            if any(pattern.search(text) for pattern in blocked):
                counts["unwanted_content"] += 1
                continue
            cleaned = corpus_filter.apply(text)
            if cleaned is None:
                continue
            language = detect_language(cleaned)
            if languages and language not in languages:
                counts["language_rejected"] += 1
                continue
            output_index = counts["accepted"]
            shard = output_index // shard_size
            path = destination / f"shard-{shard:05d}.jsonl"
            if stream is None or stream.name != str(path):
                if stream is not None:
                    stream.close()
                stream = path.open("a", encoding="utf-8")
                if str(path) not in output_files:
                    output_files.append(str(path))
            identifier = raw.get("id") or hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
            record = {"id": str(identifier), "source": str(source.get("name", kind)), "text": cleaned, "language": language}
            if isinstance(raw.get("url"), str):
                record["url"] = raw["url"]
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts["accepted"] += 1
            language_counts[language] += 1
    finally:
        if stream is not None:
            stream.close()
    manifest = {"format": "llm-corpus-v1", "pipeline": "datatrove-compatible", "source": source,
                "output_files": output_files, "counts": dict(counts), "languages": dict(language_counts),
                "filter_stats": asdict(corpus_filter.stats), "next_step": "scripts/build_token_shards.py <shards> --tokenizer <dir> --output <token-dir>"}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="YAML corpus pipeline configuration")
    args = parser.parse_args()
    config = load_yaml(args.config)
    if config.get("planning_only", False):
        parser.error("planning-only corpus profile; provide its source data and remove planning_only first")
    print(json.dumps(run_pipeline(config), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
