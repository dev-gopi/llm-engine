"""Train or resume a Vision Transformer classifier on class folders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from image_data.dataset import ImageClassificationDataset
from image_data.processor import ImageProcessor
from optim.scheduler import Scheduler
from training.checkpoint import load_checkpoint, save_checkpoint
from utils.config import load_yaml
from vision.classifier import VisionClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/vision/training.production.yaml"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--validation-data", type=Path)
    parser.add_argument("--best-output", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = load_yaml(args.config)
    torch.manual_seed(int(config.get("seed", 42)))
    dataset = ImageClassificationDataset(
        args.data or config["train_data"], int(config["image_size"]),
        processor=ImageProcessor.from_config(config, training=True),
    )
    configured_classes = int(config.get("num_classes", len(dataset.class_to_id)))
    if configured_classes != len(dataset.class_to_id):
        raise ValueError("num_classes does not match the class directories")
    config["num_classes"] = configured_classes
    device = torch.device(args.device)
    model = VisionClassifier.from_config(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 3e-4)),
                                  weight_decay=float(config.get("weight_decay", 0.05)))
    loader = DataLoader(dataset, batch_size=int(config.get("batch_size", 32)), shuffle=True,
                        num_workers=int(config.get("num_workers", 4)), pin_memory=device.type == "cuda")
    validation_path = args.validation_data or config.get("validation_data")
    validation_loader = None
    if validation_path:
        validation_dataset = ImageClassificationDataset(
            validation_path, int(config["image_size"]),
            processor=ImageProcessor.from_config(config, training=False),
        )
        if validation_dataset.class_to_id != dataset.class_to_id:
            raise ValueError("validation and training class directories must match")
        validation_loader = DataLoader(
            validation_dataset, batch_size=int(config.get("batch_size", 32)), shuffle=False,
            num_workers=int(config.get("num_workers", 4)), pin_memory=device.type == "cuda",
        )
    total_steps = max(1, int(config.get("epochs", 50)) * len(loader))
    lr_scheduler = Scheduler.from_config(optimizer, config, total_steps=total_steps)
    mixed_precision = str(config.get("mixed_precision", "bf16" if device.type == "cuda" else "none"))
    if mixed_precision not in {"none", "fp16", "bf16"}:
        raise ValueError("mixed_precision must be none, fp16, or bf16")
    if mixed_precision == "fp16" and device.type != "cuda":
        mixed_precision = "none"
    amp_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision == "fp16" and device.type == "cuda")
    step = 0
    if args.resume:
        step = load_checkpoint(args.resume, model, optimizer=optimizer, scheduler=lr_scheduler,
                               scaler=scaler, map_location=device)["step"]
    output = args.output or Path(config.get("output", "checkpoints/vision/latest.pt"))
    best_output = args.best_output or output.with_name("best.pt")
    best_validation_loss = float("inf")
    model.train()
    for epoch in range(int(config.get("epochs", 50))):
        correct = total = 0
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=mixed_precision != "none"):
                logits = model(images)
                loss = F.cross_entropy(logits, labels, label_smoothing=float(config.get("label_smoothing", 0.0)))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite vision loss at step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("max_grad_norm", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            step += 1
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.numel()
        print(f"epoch={epoch + 1} step={step} loss={loss.item():.6f} accuracy={correct / total:.4f}", flush=True)
        save_checkpoint(output, model, optimizer=optimizer, scheduler=lr_scheduler,
                        scaler=scaler, step=step,
                        metadata={"task": "vision_classification", "config": config,
                                  "class_to_id": dataset.class_to_id})
        if validation_loader is not None:
            model.eval()
            validation_loss = validation_correct = validation_total = 0.0
            with torch.inference_mode():
                for images, labels in validation_loader:
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images)
                    validation_loss += F.cross_entropy(logits.float(), labels, reduction="sum").item()
                    validation_correct += (logits.argmax(1) == labels).sum().item()
                    validation_total += labels.numel()
            validation_loss /= validation_total
            print(f"validation_loss={validation_loss:.6f} validation_accuracy={validation_correct / validation_total:.4f}")
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                save_checkpoint(best_output, model, step=step,
                                metadata={"task": "vision_classification", "config": config,
                                          "class_to_id": dataset.class_to_id,
                                          "validation_loss": validation_loss})
            model.train()
    print(output)


if __name__ == "__main__":
    main()
