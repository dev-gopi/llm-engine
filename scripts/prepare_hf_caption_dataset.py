#!/usr/bin/env python3
"""Materialize a bounded Hugging Face captioned-image subset as JSONL + PNGs."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Avoid shadowing the standard-library ``tokenize`` module with scripts/tokenize.py.
script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)


def _image(value):
    from PIL import Image

    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict) and value.get("bytes"):
        import io

        return Image.open(io.BytesIO(value["bytes"]))
    raise ValueError("image column must decode to a Pillow image or encoded bytes")


def write_split(
    dataset: str,
    split: str,
    destination: Path,
    *,
    image_column: str,
    caption_column: str,
    limit: int,
) -> int:
    from datasets import load_dataset

    if limit < 1:
        raise ValueError("split limit must be positive")
    records = load_dataset(dataset, split=split, streaming=True)
    images = destination / "images"
    images.mkdir(parents=True, exist_ok=True)
    manifest = destination / "manifest.jsonl"
    written = 0
    with manifest.open("w", encoding="utf-8") as handle:
        for row in records:
            caption = row.get(caption_column)
            if not isinstance(caption, str) or not caption.strip():
                continue
            image = _image(row.get(image_column)).convert("RGB")
            relative = Path("images") / f"{written:06d}.png"
            image.save(destination / relative, format="PNG", optimize=True)
            handle.write(
                json.dumps({"image": str(relative), "text": caption.strip()})
                + "\n"
            )
            written += 1
            if written >= limit:
                break
    if written < limit:
        raise RuntimeError(f"only wrote {written} valid samples from {dataset}/{split}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="DavidPhilips/coco2017")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--caption-column", default="caption_en")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--train-size", type=int, default=10_000)
    parser.add_argument("--validation-size", type=int, default=1_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"output is not empty: {args.output}; use --overwrite")
        shutil.rmtree(args.output)
    counts = {
        "train": write_split(
            args.dataset,
            args.train_split,
            args.output / "train",
            image_column=args.image_column,
            caption_column=args.caption_column,
            limit=args.train_size,
        ),
        "validation": write_split(
            args.dataset,
            args.validation_split,
            args.output / "validation",
            image_column=args.image_column,
            caption_column=args.caption_column,
            limit=args.validation_size,
        ),
    }
    (args.output / "source.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "license": "cc-by-4.0 (verify dataset card before redistribution)",
                "image_column": args.image_column,
                "caption_column": args.caption_column,
                "counts": counts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), **counts}, indent=2))


if __name__ == "__main__":
    main()
