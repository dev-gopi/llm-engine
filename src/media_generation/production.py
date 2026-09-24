"""Production helpers shared by audio/video generation training and serving."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def validate_generation_config(config: Mapping[str, Any], *, kind: str) -> None:
    """Fail fast on invalid or internally inconsistent media-generation configs."""
    if kind not in {"audio", "video"}:
        raise ValueError("kind must be audio or video")
    required = ["tokenizer", "train_manifest"]
    required += ["sample_rate", "duration_seconds"] if kind == "audio" else ["frames", "height", "width", "fps"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"missing {kind} generation config keys: {', '.join(missing)}")
    for key in ("batch_size", "gradient_accumulation_steps", "epochs", "timesteps"):
        if int(config.get(key, 1)) <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("learning_rate", "max_grad_norm"):
        if float(config.get(key, 1.0)) <= 0:
            raise ValueError(f"{key} must be positive")
    dropout = float(config.get("condition_dropout", 0.1))
    if not 0 <= dropout <= 1:
        raise ValueError("condition_dropout must be between zero and one")
    gamma = float(config.get("min_snr_gamma", 0.0))
    if gamma < 0:
        raise ValueError("min_snr_gamma cannot be negative")
    if kind == "video":
        factor = 2 ** int(config.get("spatial_stages", 3))
        height, width = int(config["height"]), int(config["width"])
        if height % factor or width % factor:
            raise ValueError(f"video height/width must be divisible by spatial autoencoder factor {factor}")
        channels = int(config.get("model_channels", 192))
        heads = int(config.get("temporal_heads", 8))
        cross_heads = int(config.get("cross_attention_heads", heads))
        if channels % heads or channels % cross_heads:
            raise ValueError("model_channels must be divisible by temporal and cross-attention heads")
    else:
        channels = int(config.get("model_channels", 256))
        cross_heads = int(config.get("cross_attention_heads", 8))
        if channels % cross_heads:
            raise ValueError("audio model_channels must be divisible by cross_attention_heads")
    if str(config.get("mixed_precision", "none")) not in {"none", "fp16", "bf16"}:
        raise ValueError("mixed_precision must be none, fp16, or bf16")


def configure_torch_runtime(config: Mapping[str, Any], device: torch.device) -> None:
    """Enable safe CUDA performance switches only when explicitly configured."""
    if device.type != "cuda":
        return
    if bool(config.get("allow_tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if bool(config.get("cudnn_benchmark", True)):
        torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision(str(config.get("matmul_precision", "high")))
    except (AttributeError, ValueError):
        pass


def diffusion_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    alpha_bars: Tensor,
    timesteps: Tensor,
    min_snr_gamma: float = 0.0,
) -> Tensor:
    """Per-example epsilon loss with optional Min-SNR weighting."""
    per = (prediction.float() - target.float()).square().flatten(1).mean(dim=1)
    if min_snr_gamma > 0:
        alpha = alpha_bars.to(timesteps.device)[timesteps].float().clamp(1e-8, 1 - 1e-8)
        snr = alpha / (1 - alpha)
        weights = torch.minimum(snr, torch.full_like(snr, min_snr_gamma)) / snr.clamp_min(1e-8)
        per = per * weights
    return per.mean()


def make_noise(reference: Tensor, *, generator: torch.Generator | None = None,
               noise_offset: float = 0.0, input_perturbation: float = 0.0) -> tuple[Tensor, Tensor]:
    """Return training noise and optional perturbed noise used for the noised input."""
    noise = torch.randn(reference.shape, device=reference.device, dtype=reference.dtype, generator=generator)
    if noise_offset:
        shape = (reference.shape[0], reference.shape[1]) + (1,) * (reference.ndim - 2)
        offset = torch.randn(shape, device=reference.device, dtype=reference.dtype, generator=generator)
        noise = noise + float(noise_offset) * offset
    noisy_noise = noise
    if input_perturbation:
        perturb = torch.randn(reference.shape, device=reference.device, dtype=reference.dtype, generator=generator)
        noisy_noise = noise + float(input_perturbation) * perturb
    return noise, noisy_noise


def optimizer_steps_per_epoch(loader_length: int, accumulation: int) -> int:
    if loader_length <= 0 or accumulation <= 0:
        raise ValueError("loader_length and accumulation must be positive")
    return math.ceil(loader_length / accumulation)


def manifest_paths_exist(config: Mapping[str, Any]) -> None:
    for key in ("train_manifest", "validation_manifest"):
        value = config.get(key)
        if value and not Path(value).is_file():
            raise FileNotFoundError(f"{key} not found: {value}")
