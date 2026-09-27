"""Optional native GPU low-bit/FP8 quantization adapters."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GPUQuantizationCapability:
    mode: str
    available: bool
    backend: str | None
    reason: str


def capability(mode: str) -> GPUQuantizationCapability:
    mode = mode.lower()
    if mode not in {"int4", "int8", "fp8"}:
        raise ValueError("mode must be int4, int8, or fp8")
    if not torch.cuda.is_available():
        return GPUQuantizationCapability(mode, False, None, "CUDA is unavailable")
    if importlib.util.find_spec("torchao") is None:
        return GPUQuantizationCapability(mode, False, None, "torchao is not installed")
    if mode == "fp8" and not hasattr(torch, "float8_e4m3fn"):
        return GPUQuantizationCapability(
            mode, False, "torchao", "this PyTorch build has no float8 dtype"
        )
    return GPUQuantizationCapability(mode, True, "torchao", "available")


def quantize_model(model, mode: str, **kwargs):
    cap = capability(mode)
    if not cap.available:
        raise RuntimeError(cap.reason)
    import torchao.quantization as tq

    mode = mode.lower()
    candidates = {
        "int4": ("int4_weight_only", "Int4WeightOnlyConfig"),
        "int8": ("int8_weight_only", "Int8WeightOnlyConfig"),
        "fp8": ("float8_weight_only", "Float8WeightOnlyConfig"),
    }[mode]
    fn = getattr(tq, candidates[0], None)
    if callable(fn):
        return tq.quantize_(model, fn(**kwargs) if kwargs else fn())
    cfg_cls = getattr(tq, candidates[1], None)
    if cfg_cls is not None:
        return tq.quantize_(model, cfg_cls(**kwargs))
    raise RuntimeError(
        f"installed torchao does not expose a supported {mode} quantizer"
    )


__all__ = ["GPUQuantizationCapability", "capability", "quantize_model"]
