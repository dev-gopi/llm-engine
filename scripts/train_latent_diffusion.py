#!/usr/bin/env python3
"""Train native text-conditioned latent diffusion U-Net + text encoder on a caption JSONL manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from diffusion.latent_pipeline import LatentDiffusionPipeline
from diffusion.scheduler import DiffusionScheduler
from diffusion.text_encoder import DiffusionTextEncoder
from diffusion.unet import SmallUNet
from diffusion.vae import AutoencoderKL
from image_data.dataset import CaptionedImageDataset
from media_generation.conditioning import tokenize_prompts
from tokenizer.encoder import Tokenizer
from training.checkpoint import load_checkpoint, save_checkpoint
from training.experiment_tracking import create_tracker_from_config
from utils.config import load_yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config", type=Path, default=Path("configs/diffusion/latent.production.yaml")
    )
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    tracker = create_tracker_from_config(cfg)
    if cfg.get("planning_only"):
        raise SystemExit(
            "remove planning_only after dataset/license review before training"
        )
    required = (
        "train_manifest",
        "tokenizer",
        "vae_output",
        "latent_model_output",
        "latent_text_encoder_output",
    )
    missing = [x for x in required if not cfg.get(x)]
    if missing:
        raise ValueError(f"latent diffusion config missing: {', '.join(missing)}")
    device = torch.device(args.device)
    tok = Tokenizer.load(Path(cfg["tokenizer"]))
    dataset = CaptionedImageDataset(
        cfg["train_manifest"],
        int(cfg["image_size"]),
        root=cfg.get("train_root"),
        augment=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 4)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    )
    vae = AutoencoderKL.from_config(cfg).to(device)
    load_checkpoint(
        Path(cfg["vae_output"]),
        vae,
        map_location=device,
        use_ema=True,
        restore_rng=False,
    )
    vae.eval()
    model_cfg = dict(cfg)
    model_cfg["image_channels"] = int(cfg.get("latent_channels", 4))
    model = SmallUNet.from_config(model_cfg).to(device)
    text = DiffusionTextEncoder.from_config(cfg, vocab_size=tok.vocab_size).to(device)
    params = list(model.parameters()) + list(text.parameters())
    opt = torch.optim.AdamW(
        params,
        lr=float(cfg.get("learning_rate", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )
    scheduler = DiffusionScheduler(
        int(cfg.get("timesteps", 1000)),
        float(cfg.get("beta_start", 1e-4)),
        float(cfg.get("beta_end", 2e-2)),
        device=device,
        schedule=str(cfg.get("noise_schedule", "cosine")),
    )
    pipe = LatentDiffusionPipeline(
        vae,
        model,
        scheduler,
        text,
        latent_scale=float(cfg.get("latent_scale", 0.18215)),
    )
    start = 0
    if args.resume:
        start = load_checkpoint(
            args.resume, model, optimizer=opt, map_location=device, restore_rng=True
        ).get("step", 0)
        text_resume = Path(str(args.resume) + ".text.pt")
        if text_resume.is_file():
            load_checkpoint(text_resume, text, map_location=device, restore_rng=False)
    out = Path(cfg["latent_model_output"])
    text_out = Path(cfg["latent_text_encoder_output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    text_out.parent.mkdir(parents=True, exist_ok=True)
    step = start
    epochs = int(cfg.get("epochs", 20))
    save_every = int(cfg.get("save_every_steps", 1000))
    max_grad = float(cfg.get("max_grad_norm", 1.0))
    max_len = int(cfg.get("text_max_length", 128))
    model.train()
    text.train()
    for epoch in range(epochs):
        for images, captions in loader:
            batch = tokenize_prompts(
                list(captions), tok, max_length=max_len, device=device
            )
            opt.zero_grad(set_to_none=True)
            loss = pipe.training_loss(
                images.to(device, non_blocking=True),
                token_ids=batch.ids,
                attention_mask=batch.mask,
                condition_dropout=float(cfg.get("condition_dropout", 0.1)),
                min_snr_gamma=cfg.get("min_snr_gamma"),
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite latent diffusion loss at step {step}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_grad)
            opt.step()
            step += 1
            tracker.log({"train/loss": float(loss.detach())}, step=step)
            if step % int(cfg.get("log_every_steps", 50)) == 0:
                print(
                    f"epoch={epoch + 1} step={step} loss={loss.item():.6f}", flush=True
                )
            if save_every and step % save_every == 0:
                save_checkpoint(
                    out,
                    model,
                    optimizer=opt,
                    step=step,
                    metadata={
                        "task": "latent-diffusion",
                        "config": cfg,
                        "tokenizer_fingerprint": tok.fingerprint,
                    },
                )
                save_checkpoint(
                    text_out,
                    text,
                    step=step,
                    metadata={
                        "task": "latent-diffusion-text",
                        "tokenizer_fingerprint": tok.fingerprint,
                    },
                )
    save_checkpoint(
        out,
        model,
        optimizer=opt,
        step=step,
        metadata={
            "task": "latent-diffusion",
            "config": cfg,
            "tokenizer_fingerprint": tok.fingerprint,
        },
    )
    save_checkpoint(
        text_out,
        text,
        step=step,
        metadata={
            "task": "latent-diffusion-text",
            "tokenizer_fingerprint": tok.fingerprint,
        },
    )
    print(out)
    print(text_out)
    tracker.close()


if __name__ == "__main__":
    main()
