"""Train or exactly resume the text-conditioned latent video diffusion model."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

import numpy as np
import torch
from torch.utils.data import DataLoader

from diffusion.scheduler import DiffusionScheduler
from media_generation.conditioning import build_text_conditioner, tokenize_prompts
from media_generation.production import (
    configure_torch_runtime,
    manifest_paths_exist,
    optimizer_steps_per_epoch,
    validate_generation_config,
)
from optim.ema import EMA
from optim.scheduler import Scheduler
from tokenizer.encoder import Tokenizer
from training.checkpoint import load_checkpoint, save_checkpoint
from utils.config import load_yaml
from utils.logger import configure_logging, get_logger
from video_generation.dataset import VideoCaptionDataset
from video_generation.model import VideoDiffusionModel
from video_generation.pipeline import VideoGenerationPipeline

logger = get_logger(__name__)


def _features(model, tokenizer, texts, device):
    tokens = tokenize_prompts(
        texts, tokenizer, max_length=model.text_conditioner.encoder.max_length, device=device
    )
    return model.text_conditioner.encode_features(tokens.ids, tokens.mask)


def _make_loader(dataset, config, *, batch_size: int, shuffle: bool, generator=None):
    workers = int(config.get("num_workers", 4))
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=bool(config.get("pin_memory", True)) and torch.cuda.is_available(),
        drop_last=shuffle and len(dataset) >= batch_size and bool(config.get("drop_last", True)),
        generator=generator,
    )
    if workers > 0:
        kwargs["persistent_workers"] = bool(config.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(config.get("prefetch_factor", 2))
    return DataLoader(dataset, **kwargs)


def _metadata(config, tokenizer):
    return {
        "task": "text_to_video",
        "config": config,
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "architecture_version": 2,
    }


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/video_generation/high_quality.yaml"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_yaml(args.config)
    validate_generation_config(config, kind="video")
    manifest_paths_exist(config)
    device = torch.device(args.device)
    configure_torch_runtime(config, device)
    seed = int(config.get("seed", 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    tokenizer = Tokenizer.load(Path(config["tokenizer"]))
    model = VideoDiffusionModel.from_config(
        config, text_conditioner=build_text_conditioner(config, tokenizer)
    ).to(device)
    scheduler = DiffusionScheduler(
        int(config.get("timesteps", 1000)),
        float(config.get("beta_start", 1e-4)),
        float(config.get("beta_end", 2e-2)),
        device=device,
        schedule=str(config.get("noise_schedule", "cosine")),
    )
    pipeline = VideoGenerationPipeline(model, scheduler)

    dataset = VideoCaptionDataset(
        config["train_manifest"],
        frames=int(config["frames"]),
        height=int(config["height"]),
        width=int(config["width"]),
        sampling=str(config.get("train_frame_sampling", "random_contiguous")),
        horizontal_flip_probability=float(config.get("horizontal_flip_probability", 0.5)),
    )
    loader_generator = torch.Generator()
    batch_size = int(config.get("batch_size", 1))
    loader = _make_loader(dataset, config, batch_size=batch_size, shuffle=True, generator=loader_generator)

    validation_loader = None
    if config.get("validation_manifest"):
        validation_dataset = VideoCaptionDataset(
            config["validation_manifest"],
            frames=int(config["frames"]),
            height=int(config["height"]),
            width=int(config["width"]),
            sampling=str(config.get("validation_frame_sampling", "center_contiguous")),
            horizontal_flip_probability=0.0,
        )
        validation_loader = _make_loader(
            validation_dataset,
            config,
            batch_size=int(config.get("validation_batch_size", 1)),
            shuffle=False,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 8e-5)),
        betas=(float(config.get("beta1", 0.9)), float(config.get("beta2", 0.95))),
        eps=float(config.get("adam_epsilon", 1e-8)),
        weight_decay=float(config.get("weight_decay", 0.01)),
        fused=bool(config.get("fused_optimizer", False)) and device.type == "cuda",
    )
    accumulation = int(config.get("gradient_accumulation_steps", 1))
    epochs = int(config.get("epochs", 100))
    total_steps = max(1, epochs * optimizer_steps_per_epoch(len(loader), accumulation))
    lr_scheduler = Scheduler.from_config(optimizer, config, total_steps=total_steps)
    ema = EMA(model, decay=float(config.get("ema_decay", 0.9999)))
    mixed = str(config.get("mixed_precision", "bf16" if device.type == "cuda" else "none"))
    amp_dtype = torch.bfloat16 if mixed == "bf16" else torch.float16
    amp_enabled = mixed in {"bf16", "fp16"} and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=mixed == "fp16" and device.type == "cuda")

    step = 0
    start_epoch = 0
    start_batch = 0
    best_loss = float("inf")
    if args.resume:
        state = load_checkpoint(
            args.resume,
            model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            ema=ema,
            scaler=scaler,
            map_location=device,
        )
        step = state["step"]
        trainer = state.get("trainer", {})
        start_epoch = int(trainer.get("epoch", 0))
        start_batch = int(trainer.get("next_batch", 0))
        best_loss = float(trainer.get("best_loss", float("inf")))
        print(
            f"resumed={args.resume} step={step} epoch={start_epoch + 1} next_batch={start_batch} best_loss={best_loss:.6f}",
            flush=True,
        )

    output = Path(config.get("output", "checkpoints/video_generation/latest.pt"))
    best_output = Path(config.get("best_output", "checkpoints/video_generation/best.pt"))
    metadata = _metadata(config, tokenizer)
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, epochs):
        loader_generator.manual_seed(seed + epoch)
        model.train()
        group_start = 0
        for batch_index, (videos, texts) in enumerate(loader):
            if epoch == start_epoch and batch_index < start_batch:
                continue
            if batch_index % accumulation == 0:
                group_start = batch_index
            current_group = min(accumulation, len(loader) - group_start)
            videos = videos.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                features = _features(model, tokenizer, texts, device)
                loss, metrics = pipeline.training_loss(
                    videos,
                    features.pooled,
                    context=features.context,
                    context_mask=features.mask,
                    condition_dropout=float(config.get("condition_dropout", 0.1)),
                    reconstruction_weight=float(config.get("reconstruction_weight", 0.1)),
                    min_snr_gamma=float(config.get("min_snr_gamma", 0.0)),
                    noise_offset=float(config.get("noise_offset", 0.0)),
                    input_perturbation=float(config.get("input_perturbation", 0.0)),
                )
                scaled = loss / current_group
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite video generation loss at step {step}")
            scaler.scale(scaled).backward()
            flush = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader)
            if not flush:
                continue
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config.get("max_grad_norm", 1.0))
            )
            if not torch.isfinite(torch.as_tensor(grad_norm)):
                raise FloatingPointError(f"non-finite video gradient norm at step {step}")
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()
            ema.update(model)
            step += 1
            if step % int(config.get("log_every_steps", 10)) == 0:
                print(
                    f"epoch={epoch + 1} step={step} loss={loss.item():.6f} "
                    f"diffusion={metrics['diffusion'].item():.6f} "
                    f"reconstruction={metrics['reconstruction'].item():.6f} "
                    f"grad_norm={float(grad_norm):.4f} lr={optimizer.param_groups[0]['lr']:.8g}",
                    flush=True,
                )
            if step % int(config.get("save_every_steps", 250)) == 0:
                save_checkpoint(
                    output,
                    model,
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    ema=ema,
                    scaler=scaler,
                    step=step,
                    metadata=metadata,
                    trainer={"epoch": epoch, "next_batch": batch_index + 1, "best_loss": best_loss},
                )

        if validation_loader is not None:
            model.eval()
            total = 0.0
            count = 0
            validation_generator = torch.Generator(device=device)
            validation_generator.manual_seed(int(config.get("validation_seed", 12345)))
            with ema.average_parameters(model), torch.inference_mode():
                for videos, texts in validation_loader:
                    videos = videos.to(device, non_blocking=True)
                    features = _features(model, tokenizer, texts, device)
                    loss, _ = pipeline.training_loss(
                        videos,
                        features.pooled,
                        context=features.context,
                        context_mask=features.mask,
                        condition_dropout=0.0,
                        reconstruction_weight=float(config.get("reconstruction_weight", 0.1)),
                        min_snr_gamma=float(config.get("min_snr_gamma", 0.0)),
                        generator=validation_generator,
                    )
                    total += loss.item(); count += 1
            validation_loss = total / max(count, 1)
            print(f"epoch={epoch + 1} validation_loss={validation_loss:.6f}", flush=True)
            if validation_loss < best_loss:
                best_loss = validation_loss
                with ema.average_parameters(model):
                    save_checkpoint(
                        best_output,
                        model,
                        step=step,
                        metadata={**metadata, "validation_loss": validation_loss, "inference_only": True},
                    )

        save_checkpoint(
            output,
            model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            ema=ema,
            scaler=scaler,
            step=step,
            metadata=metadata,
            trainer={"epoch": epoch + 1, "next_batch": 0, "best_loss": best_loss},
        )
        start_batch = 0

    print(output)


if __name__ == "__main__":
    main()
