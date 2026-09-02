"""Sample a scratch diffusion checkpoint, or smoke-test an untrained model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

import torch

from diffusion.pipeline import DiffusionPipeline
from diffusion.scheduler import DiffusionScheduler
from diffusion.unet import SmallUNet
from image_data.processor import tensor_to_image
from utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/diffusion/model.small.yaml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/generated_images/sample.png"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = load_yaml(args.config)
    device = torch.device(args.device)
    model = SmallUNet.from_config(config).to(device)
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(payload.get("model", payload))
    scheduler = DiffusionScheduler(
        timesteps=int(config.get("timesteps", 100)),
        beta_start=float(config.get("beta_start", 1e-4)),
        beta_end=float(config.get("beta_end", 2e-2)),
        device=device,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    sample = DiffusionPipeline(model, scheduler).sample(
        1, int(config["image_size"]), device=device, generator=generator
    )[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_image(sample).save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
