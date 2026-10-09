#!/usr/bin/env python3
"""Generate images from a prompt using native latent diffusion checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

import torch

from diffusion.latent_pipeline import LatentDiffusionPipeline
from diffusion.scheduler import DiffusionScheduler
from diffusion.text_encoder import DiffusionTextEncoder
from diffusion.unet import SmallUNet
from diffusion.vae import AutoencoderKL
from image_data.processor import tensor_to_image
from media_generation.conditioning import tokenize_prompts
from tokenizer.encoder import Tokenizer
from training.checkpoint import load_checkpoint
from utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/diffusion/latent.local_4gb.yaml")
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/generated_images/prompt.png"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda was requested but CUDA is unavailable")
    tok = Tokenizer.load(Path(cfg["tokenizer"]))
    vae = AutoencoderKL.from_config(cfg).to(device)
    load_checkpoint(Path(cfg["vae_output"]), vae, map_location=device, use_ema=True, restore_rng=False)
    model_cfg = dict(cfg)
    model_cfg["image_channels"] = int(cfg["latent_channels"])
    model = SmallUNet.from_config(model_cfg).to(device)
    load_checkpoint(Path(cfg["latent_model_output"]), model, map_location=device, restore_rng=False)
    text = DiffusionTextEncoder.from_config(cfg, vocab_size=tok.vocab_size).to(device)
    load_checkpoint(Path(cfg["latent_text_encoder_output"]), text, map_location=device, restore_rng=False)
    scheduler = DiffusionScheduler(
        int(cfg["timesteps"]),
        float(cfg.get("beta_start", 1e-4)),
        float(cfg.get("beta_end", 2e-2)),
        device=device,
        schedule=str(cfg.get("noise_schedule", "cosine")),
    )
    length = int(cfg.get("text_max_length", 64))
    if args.negative_prompt:
        # Tokenize together so both tensors share the same padded sequence
        # length, as classifier-free guidance requires.
        conditions = tokenize_prompts(
            [args.prompt, args.negative_prompt], tok, max_length=length, device=device
        )
        prompt_ids, negative_ids = conditions.ids.split(1)
        prompt_mask, negative_mask = conditions.mask.split(1)
    else:
        conditions = tokenize_prompts([args.prompt], tok, max_length=length, device=device)
        prompt_ids, prompt_mask = conditions.ids, conditions.mask
        negative_ids = negative_mask = None
    image = LatentDiffusionPipeline(vae, model, scheduler, text, latent_scale=float(cfg["latent_scale"])).sample(
        1,
        int(cfg["image_size"]),
        device=device,
        token_ids=prompt_ids,
        attention_mask=prompt_mask,
        negative_token_ids=negative_ids,
        negative_attention_mask=negative_mask,
        guidance_scale=args.guidance_scale or float(cfg.get("guidance_scale", 5.0)),
        inference_steps=args.steps or int(cfg.get("inference_steps", 50)),
        generator=torch.Generator(device=device).manual_seed(args.seed),
    )[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_image(image).save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
