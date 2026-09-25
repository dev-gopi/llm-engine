"""Safe opt-in model quantization helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from utils.logger import get_logger

logger = get_logger(__name__)

Q1_0_GROUP_SIZE = 128
Q1_0_EFFECTIVE_BITS = 1.0 + 16.0 / Q1_0_GROUP_SIZE


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


def quantize_q1_0(
    tensor: Tensor,
    *,
    group_size: int = Q1_0_GROUP_SIZE,
) -> tuple[Tensor, Tensor, int]:
    """Pack a floating tensor into a group-wise binary Q1_0 representation.

    Every weight is represented by one sign bit and each group shares one FP16
    scale. With the llama.cpp-compatible 128-value group size this is exactly
    1.125 effective bits/weight before container metadata. The scale is the
    mean absolute value, which minimizes squared error for a fixed binary sign
    pattern. This helper is an engine-native research/export representation;
    it does not claim GGUF binary compatibility by itself.
    """
    if not tensor.is_floating_point():
        raise ValueError("Q1_0 quantization requires a floating-point tensor")
    if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size < 8 or group_size % 8:
        raise ValueError("Q1_0 group_size must be a positive multiple of 8")
    values = tensor.detach().to(torch.float32).reshape(-1)
    original_numel = values.numel()
    if original_numel == 0:
        raise ValueError("Q1_0 quantization does not support empty tensors")
    groups = (original_numel + group_size - 1) // group_size
    padded_numel = groups * group_size
    if padded_numel != original_numel:
        values = torch.cat((values, torch.zeros(padded_numel - original_numel, device=values.device)))
    values = values.view(groups, group_size)
    scales = values.abs().mean(dim=1).clamp_min(torch.finfo(torch.float16).tiny).to(torch.float16)
    signs = (values >= 0).to(torch.uint8).view(groups, group_size // 8, 8)
    shifts = torch.arange(8, device=values.device, dtype=torch.int64)
    packed = torch.sum(signs.to(torch.int64) << shifts, dim=-1).to(torch.uint8)
    return packed.cpu().contiguous(), scales.cpu().contiguous(), original_numel


def dequantize_q1_0(
    packed: Tensor,
    scales: Tensor,
    *,
    shape: torch.Size | tuple[int, ...],
    original_numel: int,
    group_size: int = Q1_0_GROUP_SIZE,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Restore a tensor produced by :func:`quantize_q1_0`."""
    if packed.dtype != torch.uint8 or scales.dtype not in {torch.float16, torch.float32, torch.bfloat16}:
        raise ValueError("invalid Q1_0 packed tensor or scales")
    if not isinstance(group_size, int) or group_size < 8 or group_size % 8:
        raise ValueError("Q1_0 group_size must be a positive multiple of 8")
    expected_numel = int(torch.tensor(shape).prod().item())
    if original_numel != expected_numel:
        raise ValueError("Q1_0 metadata does not match tensor shape")
    groups = (original_numel + group_size - 1) // group_size
    if tuple(packed.shape) != (groups, group_size // 8) or scales.numel() != groups:
        raise ValueError("Q1_0 metadata does not match packed storage")
    shifts = torch.arange(8, device=packed.device, dtype=torch.int64)
    bits = ((packed.to(torch.int64).unsqueeze(-1) >> shifts) & 1).reshape(groups, group_size)
    signs = bits.to(torch.float32).mul_(2.0).sub_(1.0)
    restored = signs * scales.to(device=packed.device, dtype=torch.float32).reshape(-1, 1)
    return restored.reshape(-1)[:original_numel].reshape(shape).to(dtype)


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

@dataclass(frozen=True)
class QuantizedDeploymentManifest:
    """Interchange contract for GPTQ/AWQ/GGUF-compatible artifacts."""
    schema_version: int
    format: str
    architecture: str
    model_config_fingerprint: str
    calibration_sha256: str
    calibration_tokens: int
    source_dtype: str
    quantization_bits: int
    quality: dict
    latency: dict
    memory: dict

    def to_dict(self):
        from dataclasses import asdict
        return asdict(self)


def architecture_fingerprint(config: dict) -> str:
    import hashlib
    import json
    keys=("architecture","vocab_size","hidden_size","layers","heads","kv_heads","ffn_hidden_size","position_type","max_position")
    payload={k:config.get(k) for k in keys}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",", ":")).encode()).hexdigest()


def validate_quantized_manifest(manifest: dict, *, config: dict) -> None:
    required=("schema_version","format","architecture","model_config_fingerprint","calibration_sha256","calibration_tokens","quantization_bits")
    missing=[k for k in required if k not in manifest]
    if missing: raise ValueError(f"quantized manifest missing: {', '.join(missing)}")
    if manifest["schema_version"] != 1: raise ValueError("unsupported quantized manifest schema")
    if manifest["format"] not in {"gptq","awq","gguf"}: raise ValueError("unsupported quantized format")
    if manifest["architecture"] != "MiniGPT": raise ValueError("architecture is not MiniGPT")
    if manifest["model_config_fingerprint"] != architecture_fingerprint(config): raise ValueError("quantized artifact architecture is incompatible with model config")
    if int(manifest["calibration_tokens"]) < 1 or len(str(manifest["calibration_sha256"])) != 64: raise ValueError("invalid calibration provenance")
    if int(manifest["quantization_bits"]) not in {4,8}: raise ValueError("quantization bits must be 4 or 8")


def build_quantized_manifest(*, fmt: str, config: dict, calibration_sha256: str, calibration_tokens: int, bits: int, quality: dict | None = None, latency: dict | None = None, memory: dict | None = None) -> dict:
    manifest=QuantizedDeploymentManifest(1,fmt,"MiniGPT",architecture_fingerprint(config),calibration_sha256,int(calibration_tokens),"float32",int(bits),quality or {},latency or {},memory or {})
    validate_quantized_manifest(manifest.to_dict(), config=config)
    return manifest.to_dict()
