"""Build the scratch vision encoder and print its output and parameter sizes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

import torch

from utils.config import load_yaml
from vision.encoder import VisionEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/vision/model.small.yaml"))
    args = parser.parse_args()
    config = load_yaml(args.config)
    model = VisionEncoder.from_config(config).eval()
    image_size = int(config["image_size"])
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, image_size, image_size))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"parameters={parameters:,}")
    print(f"output_shape={tuple(output.shape)}")


if __name__ == "__main__":
    main()
