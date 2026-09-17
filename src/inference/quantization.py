"""Safe opt-in model quantization helpers."""

from __future__ import annotations

import torch
from torch import nn


def prepare_model_for_inference(
    model: nn.Module, *, device: torch.device, weight_dtype: str = "float32",
    quantization: str = "none",
) -> nn.Module:
    """Apply config-selected precision and optional CPU quantization."""
    name = str(weight_dtype).lower()
    quantization = str(quantization).lower()
    dtypes = {"float32": torch.float32, "float16": torch.float16,
              "bfloat16": torch.bfloat16}
    if name not in dtypes:
        raise ValueError("weight_dtype must be float32, float16, or bfloat16")
    if quantization not in {"none", "int8_dynamic"}:
        raise ValueError("quantization must be none or int8_dynamic")
    if device.type == "cpu" and name == "float16":
        raise ValueError("float16 CPU inference is unsupported; use bfloat16 or int8_dynamic")
    model.to(device=device, dtype=dtypes[name]).eval()
    if quantization == "int8_dynamic":
        if device.type != "cpu":
            raise ValueError("int8_dynamic quantization requires device: cpu")
        model = quantize_dynamic_cpu(model)
    return model


def quantize_dynamic_cpu(model: nn.Module, *, dtype: torch.dtype = torch.qint8) -> nn.Module:
    """Quantize Linear layers for CPU inference; never mutates the source model."""
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("dynamic quantization is CPU-only; move the model to CPU first")
    if dtype not in {torch.qint8, torch.float16}:
        raise ValueError("dynamic quantization dtype must be qint8 or float16")
    return torch.ao.quantization.quantize_dynamic(model.eval(), {nn.Linear}, dtype=dtype, inplace=False)
