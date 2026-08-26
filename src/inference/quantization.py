"""Safe opt-in model quantization helpers."""

from __future__ import annotations

import torch
from torch import nn


def quantize_dynamic_cpu(model: nn.Module, *, dtype: torch.dtype = torch.qint8) -> nn.Module:
    """Quantize Linear layers for CPU inference; never mutates the source model."""
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("dynamic quantization is CPU-only; move the model to CPU first")
    if dtype not in {torch.qint8, torch.float16}:
        raise ValueError("dynamic quantization dtype must be qint8 or float16")
    return torch.ao.quantization.quantize_dynamic(model.eval(), {nn.Linear}, dtype=dtype, inplace=False)
