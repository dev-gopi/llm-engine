"""Train or resume the scratch diffusion model on an image directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

import torch
from torch.utils.data import DataLoader

from diffusion.pipeline import DiffusionPipeline
from diffusion.scheduler import DiffusionScheduler
from diffusion.unet import SmallUNet
from image_data.dataset import ImageClassificationDataset, ImageDataset
from image_data.processor import tensor_to_image
from image_data.processor import ImageProcessor
from optim.ema import EMA
from optim.scheduler import Scheduler
from training.checkpoint import load_checkpoint, save_checkpoint
from utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/diffusion/training.production.yaml"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--validation-data", type=Path)
    parser.add_argument("--best-output", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = load_yaml(args.config)
    torch.manual_seed(int(config.get("seed", 42)))
    device = torch.device(args.device)
    model = SmallUNet.from_config(config).to(device)
    scheduler = DiffusionScheduler(
        int(config.get("timesteps", 1000)), float(config.get("beta_start", 1e-4)),
        float(config.get("beta_end", 2e-2)), device=device,
        schedule=str(config.get("noise_schedule", "cosine")),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config.get("learning_rate", 2e-4)),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    num_classes = config.get("num_classes")
    dataset_type = ImageClassificationDataset if num_classes is not None else ImageDataset
    dataset = dataset_type(
        args.data or config["train_data"], int(config["image_size"]),
        processor=ImageProcessor.from_config(config, training=True),
    )
    if num_classes is not None and len(dataset.class_to_id) != int(num_classes):
        raise ValueError("num_classes does not match the diffusion class directories")
    loader = DataLoader(dataset, batch_size=int(config.get("batch_size", 16)), shuffle=True,
                        num_workers=int(config.get("num_workers", 4)), pin_memory=device.type == "cuda",
                        drop_last=len(dataset) >= int(config.get("batch_size", 16)))
    validation_path = args.validation_data or config.get("validation_data")
    validation_loader = None
    if validation_path:
        validation_dataset = dataset_type(
            validation_path, int(config["image_size"]),
            processor=ImageProcessor.from_config(config, training=False),
        )
        if (num_classes is not None and
                validation_dataset.class_to_id != dataset.class_to_id):
            raise ValueError("validation and training class directories must match")
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=int(config.get("batch_size", 16)), shuffle=False,
            num_workers=int(config.get("num_workers", 4)), pin_memory=device.type == "cuda",
        )
    total_steps = max(1, int(config.get("epochs", 100)) * len(loader))
    lr_scheduler = Scheduler.from_config(optimizer, config, total_steps=total_steps)
    ema = EMA(model, decay=float(config.get("ema_decay", 0.9999)))
    mixed_precision = str(config.get("mixed_precision", "bf16" if device.type == "cuda" else "none"))
    if mixed_precision not in {"none", "fp16", "bf16"}:
        raise ValueError("mixed_precision must be none, fp16, or bf16")
    if mixed_precision == "fp16" and device.type != "cuda":
        mixed_precision = "none"
    amp_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision == "fp16" and device.type == "cuda")
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer=optimizer, scheduler=lr_scheduler,
                                     ema=ema, scaler=scaler, map_location=device)["step"]
    pipeline = DiffusionPipeline(model, scheduler)
    output = args.output or Path(config.get("output", "checkpoints/diffusion/latest.pt"))
    best_output = args.best_output or output.with_name("best.pt")
    best_validation_loss = float("inf")
    epochs = int(config.get("epochs", 100))
    save_every = int(config.get("save_every_steps", 1000))
    sample_every = int(config.get("sample_every_steps", 0))
    sample_directory = Path(config.get("sample_output_dir", "outputs/generated_images/training"))
    step = start_step
    model.train()
    for epoch in range(epochs):
        for batch in loader:
            images, class_labels = batch if num_classes is not None else (batch, None)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=mixed_precision != "none"):
                loss = pipeline.training_loss(
                    images.to(device, non_blocking=True),
                    class_labels=(class_labels.to(device) if class_labels is not None else None),
                    condition_dropout=float(config.get("condition_dropout", 0.1)),
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite diffusion loss at step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("max_grad_norm", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            ema.update(model)
            step += 1
            if step % int(config.get("log_every_steps", 50)) == 0:
                print(f"epoch={epoch + 1} step={step} loss={loss.item():.6f}", flush=True)
            if save_every > 0 and step % save_every == 0:
                save_checkpoint(output, model, optimizer=optimizer, scheduler=lr_scheduler,
                                ema=ema, scaler=scaler, step=step,
                                metadata={"task": "diffusion", "config": config})
            if sample_every > 0 and step % sample_every == 0:
                preview_labels = (
                    torch.tensor([step % int(num_classes)], device=device)
                    if num_classes is not None else None
                )
                with ema.average_parameters(model):
                    preview = pipeline.sample(
                        1, int(config["image_size"]), device=device,
                        class_labels=preview_labels,
                        guidance_scale=float(config.get("guidance_scale", 3.0)),
                        inference_steps=int(config.get("inference_steps", scheduler.timesteps)),
                    )[0]
                sample_directory.mkdir(parents=True, exist_ok=True)
                tensor_to_image(preview).save(sample_directory / f"step-{step:08d}.png")
                model.train()
        if validation_loader is not None:
            model.eval()
            validation_loss = validation_batches = 0.0
            validation_generator = torch.Generator(device=device).manual_seed(
                int(config.get("validation_seed", 1234))
            )
            with ema.average_parameters(model), torch.inference_mode():
                for batch in validation_loader:
                    images, class_labels = batch if num_classes is not None else (batch, None)
                    validation_loss += pipeline.training_loss(
                        images.to(device), generator=validation_generator,
                        class_labels=(class_labels.to(device) if class_labels is not None else None),
                    ).item()
                    validation_batches += 1
            validation_loss /= validation_batches
            print(f"epoch={epoch + 1} validation_loss={validation_loss:.6f}")
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                save_checkpoint(best_output, model, ema=ema, step=step,
                                metadata={"task": "diffusion", "config": config,
                                          "validation_loss": validation_loss})
            model.train()
    save_checkpoint(output, model, optimizer=optimizer, scheduler=lr_scheduler,
                    ema=ema, scaler=scaler, step=step,
                    metadata={"task": "diffusion", "config": config})
    print(output)


if __name__ == "__main__":
    main()
