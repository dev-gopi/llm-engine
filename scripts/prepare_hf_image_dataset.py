"""Download a bounded Hugging Face image dataset into class folders."""

from __future__ import annotations

import argparse
import io
import json
import shutil
import tempfile
import urllib.request
import sys
from pathlib import Path
from typing import Any

# Avoid shadowing the standard-library ``tokenize`` module with scripts/tokenize.py.
script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

import pyarrow.parquet as pq


MNIST_LABELS = tuple(str(index) for index in range(10))


def parquet_urls(dataset: str, config: str, split: str, timeout: float = 60.0) -> list[str]:
    url = f"https://huggingface.co/api/datasets/{dataset}/parquet/{config}/{split}"
    request = urllib.request.Request(url, headers={"User-Agent": "LLMEngine/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        values = json.loads(response.read().decode("utf-8"))
    if not isinstance(values, list) or not values:
        raise ValueError(f"Hugging Face returned no Parquet files for {dataset}/{config}/{split}")
    return [str(value) for value in values]


def image_bytes(value: Any) -> bytes:
    if isinstance(value, dict) and isinstance(value.get("bytes"), bytes):
        return value["bytes"]
    if isinstance(value, bytes):
        return value
    raise ValueError("image column must contain encoded bytes")


def prepare_split(
    dataset: str, config: str, source_split: str, destination: Path, *,
    image_column: str, label_column: str, labels: tuple[str, ...], limit: int,
    timeout: float,
) -> int:
    from PIL import Image

    if limit < 1:
        raise ValueError("split limit must be positive")
    destination.mkdir(parents=True, exist_ok=True)
    written = 0
    for url in parquet_urls(dataset, config, source_split, timeout):
        if written >= limit:
            break
        with tempfile.NamedTemporaryFile(suffix=".parquet") as temporary:
            request = urllib.request.Request(url, headers={"User-Agent": "LLMEngine/1.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                shutil.copyfileobj(response, temporary)
            temporary.flush()
            parquet = pq.ParquetFile(temporary.name)
            for batch in parquet.iter_batches(columns=[image_column, label_column]):
                for row in batch.to_pylist():
                    label_id = int(row[label_column])
                    if not 0 <= label_id < len(labels):
                        raise ValueError(f"label ID {label_id} is outside configured label names")
                    class_directory = destination / labels[label_id]
                    class_directory.mkdir(parents=True, exist_ok=True)
                    output = class_directory / f"{written:06d}.png"
                    with Image.open(io.BytesIO(image_bytes(row[image_column]))) as image:
                        image.convert("RGB").save(output, format="PNG")
                    written += 1
                    if written >= limit:
                        break
                if written >= limit:
                    break
    if written < limit:
        raise ValueError(f"requested {limit} images but Hugging Face supplied only {written}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ylecun/mnist")
    parser.add_argument("--dataset-config", default="mnist")
    parser.add_argument("--output", type=Path, default=Path("data/images/hf-mnist"))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="test")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--labels", default=",".join(MNIST_LABELS),
                        help="comma-separated class names ordered by numeric label ID")
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--validation-size", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    labels = tuple(label.strip() for label in args.labels.split(",") if label.strip())
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("labels must contain at least two unique class names")
    if args.output.exists() and any(args.output.rglob("*.png")):
        if not args.overwrite:
            raise FileExistsError(f"output already contains images: {args.output}; pass --overwrite")
        shutil.rmtree(args.output)
    counts = {
        "train": prepare_split(
            args.dataset, args.dataset_config, args.train_split, args.output / "train",
            image_column=args.image_column, label_column=args.label_column, labels=labels,
            limit=args.train_size, timeout=args.timeout,
        ),
        "validation": prepare_split(
            args.dataset, args.dataset_config, args.validation_split, args.output / "validation",
            image_column=args.image_column, label_column=args.label_column, labels=labels,
            limit=args.validation_size, timeout=args.timeout,
        ),
    }
    manifest = {
        "source": args.dataset, "config": args.dataset_config,
        "license": "MIT for the default ylecun/mnist dataset; verify when overriding",
        "labels": list(labels), "counts": counts,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), **manifest}, indent=2))


if __name__ == "__main__":
    main()
