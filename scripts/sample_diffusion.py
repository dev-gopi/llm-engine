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
from training.checkpoint import load_checkpoint
from utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/diffusion/model.small.yaml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/generated_images/sample.png"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--class-id", type=int)
    parser.add_argument("--guidance-scale", type=float)
    args = parser.parse_args()
    config = load_yaml(args.config)
    device = torch.device(args.device)
    model = SmallUNet.from_config(config).to(device)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, map_location=device, use_ema=True,
                        restore_rng=False)
    scheduler = DiffusionScheduler(
        timesteps=int(config.get("timesteps", 100)),
        beta_start=float(config.get("beta_start", 1e-4)),
        beta_end=float(config.get("beta_end", 2e-2)),
        device=device,
        schedule=str(config.get("noise_schedule", "linear")),
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    class_labels = None
    if args.class_id is not None:
        if model.num_classes is None:
            parser.error("--class-id requires num_classes in the model config")
        if not 0 <= args.class_id < model.num_classes:
            parser.error(f"--class-id must be between 0 and {model.num_classes - 1}")
        class_labels = torch.full((args.batch_size,), args.class_id, device=device, dtype=torch.long)
    sample = DiffusionPipeline(model, scheduler).sample(
        args.batch_size, int(config["image_size"]), device=device, generator=generator,
        inference_steps=args.steps or config.get("inference_steps"), eta=args.eta,
        class_labels=class_labels,
        guidance_scale=(args.guidance_scale if args.guidance_scale is not None
                        else float(config.get("guidance_scale", 1.0))),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(sample):
        output = args.output if args.batch_size == 1 else args.output.with_stem(f"{args.output.stem}-{index:03d}")
        tensor_to_image(image).save(output)
        print(output)


if __name__ == "__main__":
    main()
