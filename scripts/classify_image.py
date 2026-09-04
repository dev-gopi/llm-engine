"""Classify one image with a trained vision checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

import torch

from image_data.processor import ImageProcessor
from training.checkpoint import load_checkpoint
from utils.config import load_yaml
from vision.classifier import VisionClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/vision/training.production.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = load_yaml(args.config)
    device = torch.device(args.device)
    model = VisionClassifier.from_config(config).to(device).eval()
    info = load_checkpoint(args.checkpoint, model, map_location=device, restore_rng=False)
    labels = info["metadata"].get("class_to_id", {})
    id_to_class = {identifier: name for name, identifier in labels.items()}
    image = ImageProcessor.from_config(config)(args.image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(image).softmax(-1)[0]
    values, identifiers = probabilities.topk(min(args.top_k, probabilities.numel()))
    for value, identifier in zip(values.tolist(), identifiers.tolist(), strict=True):
        print(f"{id_to_class.get(identifier, str(identifier))}\t{value:.6f}")


if __name__ == "__main__":
    main()
