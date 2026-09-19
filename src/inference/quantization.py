"""Safe opt-in model quantization helpers."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def quantize_int4(tensor: Tensor) -> tuple[Tensor, Tensor, int]:
    """Symmetrically quantize a floating tensor and pack two signed INT4s/byte.

    The returned scale is per tensor, which keeps the representation portable
    across PyTorch and safetensors consumers.  ``original_numel`` is required
    because an odd-sized tensor receives one padding nibble.
    """
    if not tensor.is_floating_point():
        raise ValueError("INT4 quantization requires a floating-point tensor")
    values = tensor.detach().to(torch.float32).reshape(-1)
    scale = values.abs().amax().reshape(1) / 7
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    quantized = torch.clamp(torch.round(values / scale), -8, 7).to(torch.int16) + 8
    if quantized.numel() % 2:
        quantized = torch.cat((quantized, torch.zeros(1, device=quantized.device, dtype=quantized.dtype)))
    packed = (quantized[0::2] | (quantized[1::2] << 4)).to(torch.uint8)
    return packed.cpu(), scale.cpu(), values.numel()


def dequantize_int4(
    packed: Tensor, scale: Tensor, *, shape: torch.Size | tuple[int, ...], original_numel: int,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Restore a tensor produced by :func:`quantize_int4`."""
    if packed.dtype != torch.uint8 or scale.numel() != 1:
        raise ValueError("invalid packed INT4 tensor or scale")
    nibbles = torch.stack((packed.to(torch.int16) & 0x0F, packed.to(torch.int16) >> 4), dim=1).reshape(-1)
    if original_numel != int(torch.tensor(shape).prod()) or original_numel > nibbles.numel():
        raise ValueError("INT4 metadata does not match the packed tensor")
    return ((nibbles[:original_numel] - 8).to(torch.float32) * scale.to(torch.float32)).reshape(shape).to(dtype)


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
    if quantization == "int8_dynamic" and device.type != "cpu":
        raise ValueError("int8_dynamic quantization requires device: cpu")
    if quantization == "int8_dynamic" and name != "float32":
        raise ValueError("int8_dynamic quantization requires weight_dtype: float32")
    model.to(device=device, dtype=dtypes[name]).eval()
    if quantization == "int8_dynamic":
        model = quantize_dynamic_cpu(model)
    return model


def quantize_dynamic_cpu(model: nn.Module, *, dtype: torch.dtype = torch.qint8) -> nn.Module:
    """Quantize Linear layers for CPU inference; never mutates the source model."""
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("dynamic quantization is CPU-only; move the model to CPU first")
    if dtype not in {torch.qint8, torch.float16}:
        raise ValueError("dynamic quantization dtype must be qint8 or float16")
    return torch.ao.quantization.quantize_dynamic(model.eval(), {nn.Linear}, dtype=dtype, inplace=False)
