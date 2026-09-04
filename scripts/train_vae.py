"""Train or resume the native image VAE used by latent diffusion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

import torch
from torch.utils.data import DataLoader

from diffusion.vae import AutoencoderKL
from image_data.dataset import ImageDataset
from image_data.processor import ImageProcessor
from optim.scheduler import Scheduler
from training.checkpoint import load_checkpoint, save_checkpoint
from utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/diffusion/latent.production.yaml"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = load_yaml(args.config)
    torch.manual_seed(int(config.get("seed", 42)))
    device = torch.device(args.device)
    model = AutoencoderKL.from_config(config).to(device)
    dataset = ImageDataset(
        args.data or config["train_data"], int(config["image_size"]),
        processor=ImageProcessor.from_config(config, training=True),
    )
    loader = DataLoader(dataset, batch_size=int(config.get("vae_batch_size", 16)), shuffle=True,
                        num_workers=int(config.get("num_workers", 4)), pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("vae_learning_rate", 1e-4)),
                                  weight_decay=float(config.get("vae_weight_decay", 0.01)))
    scheduler = Scheduler.from_config(
        optimizer, config, total_steps=max(1, int(config.get("vae_epochs", 50)) * len(loader))
    )
    mixed_precision = str(config.get("mixed_precision", "none"))
    dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision == "fp16" and device.type == "cuda")
    step = 0
    if args.resume:
        step = load_checkpoint(args.resume, model, optimizer=optimizer, scheduler=scheduler,
                               scaler=scaler, map_location=device)["step"]
    output = args.output or Path(config.get("vae_output", "checkpoints/diffusion/vae.pt"))
    model.train()
    for epoch in range(int(config.get("vae_epochs", 50))):
        for images in loader:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=dtype,
                                enabled=mixed_precision != "none"):
                result = model(images)
                loss = model.loss(result, images, kl_weight=float(config.get("kl_weight", 1e-6)))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite VAE loss at step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("max_grad_norm", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
        print(f"epoch={epoch + 1} step={step} vae_loss={loss.item():.6f}", flush=True)
        save_checkpoint(output, model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                        step=step, metadata={"task": "vae", "config": config})
    print(output)


if __name__ == "__main__":
    main()
