"""Report optional accelerator features without making them hard dependencies."""

from __future__ import annotations

import torch


def training_capabilities() -> dict[str, bool | str | int]:
    cuda = torch.cuda.is_available()
    major = minor = 0
    if cuda:
        major, minor = torch.cuda.get_device_capability()
    return {
        "cuda": cuda,
        "cuda_device_count": torch.cuda.device_count() if cuda else 0,
        "bf16": bool(cuda and torch.cuda.is_bf16_supported()),
        "fp16": cuda,
        # Native float8 dtypes exist before every GPU/backend can execute FP8 kernels.
        "fp8_dtype": hasattr(torch, "float8_e4m3fn"),
        "fp8_hardware": bool(cuda and major >= 9),
        "fused_adamw": bool(cuda),
        "sdpa": hasattr(torch.nn.functional, "scaled_dot_product_attention"),
        "flash_sdpa_hardware": bool(cuda and major >= 8),
        "compute_capability": f"{major}.{minor}" if cuda else "none",
    }
