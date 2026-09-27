"""Native quantized-format capability and export contracts.

GGUF is implemented as a real GGUF v3 binary container. GPTQ/AWQ remain
provider-specific because their packed-kernel layouts differ by runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import struct
from typing import Any, Mapping


class NativeFormatUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class FormatCapability:
    format: str
    available: bool
    dependency: str | None
    reason: str


def capabilities() -> dict[str, FormatCapability]:
    checks = {
        "gptq": ("auto_gptq", "AutoGPTQ/GPTQ runtime"),
        "awq": ("awq", "AWQ runtime"),
        "gguf": ("gguf", "GGUF Python runtime"),
    }
    out: dict[str, FormatCapability] = {}
    for fmt, (module, label) in checks.items():
        if fmt == "gguf":
            out[fmt] = FormatCapability(fmt, True, None, "built-in GGUF v3 writer available")
            continue
        ok = importlib.util.find_spec(module) is not None
        out[fmt] = FormatCapability(fmt, ok, module, "available" if ok else f"{label} is not installed")
    return out


def require_native(fmt: str) -> FormatCapability:
    fmt = fmt.lower()
    cap = capabilities().get(fmt)
    if cap is None:
        raise ValueError(f"unsupported native format: {fmt}")
    if not cap.available:
        raise NativeFormatUnavailable(cap.reason)
    return cap


# GGUF constants from the public binary format. We intentionally write only
# scalar metadata and F32/F16 tensors, which makes this exporter deterministic
# and portable; a runtime can subsequently quantize the tensors using its own
# kernel-specific implementation.
_GGUF_MAGIC = b"GGUF"
_GGUF_VERSION = 3
_GGML_F32 = 0
_GGML_F16 = 1
_KV_STRING = 8
_KV_UINT32 = 4
_KV_BOOL = 7


def _gguf_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv(key: str, value: str | int | bool) -> bytes:
    out = _gguf_string(key)
    if isinstance(value, bool):
        return out + struct.pack("<I<?", _KV_BOOL, value)
    if isinstance(value, int):
        return out + struct.pack("<II", _KV_UINT32, value)
    return out + struct.pack("<I", _KV_STRING) + _gguf_string(value)


def _tensor_bytes(tensor: Any) -> tuple[int, tuple[int, ...], bytes]:
    import torch
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("GGUF export expects torch.Tensor values")
    if tensor.layout != torch.strided:
        raise ValueError("GGUF export does not support sparse tensors")
    value = tensor.detach().contiguous().cpu()
    if value.dtype == torch.float16:
        return _GGML_F16, tuple(reversed(value.shape)), value.numpy().tobytes(order="C")
    if value.dtype in (torch.float32, torch.bfloat16):
        if value.dtype == torch.bfloat16:
            value = value.float()
        return _GGML_F32, tuple(reversed(value.shape)), value.numpy().tobytes(order="C")
    raise ValueError(f"unsupported GGUF tensor dtype: {value.dtype}; convert to float16/float32 first")


def export_gguf(
    state_dict: Mapping[str, Any],
    output: str | Path,
    *,
    architecture: str = "llm-engine",
    model_name: str = "llm-engine",
    metadata: Mapping[str, str | int | bool] | None = None,
    alignment: int = 32,
) -> Path:
    """Write a deterministic GGUF v3 container from a torch state dict.

    This is a container/export implementation, not a claim that arbitrary model
    architectures are loadable by llama.cpp. Architecture-specific loader metadata
    must be supplied by the target runtime.
    """
    if alignment < 1 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    tensors = []
    for name in sorted(state_dict):
        dtype, shape, raw = _tensor_bytes(state_dict[name])
        tensors.append((name, dtype, shape, raw))
    kv = {"general.architecture": architecture, "general.name": model_name}
    if metadata:
        kv.update(metadata)
    kv_bytes = b"".join(_kv(k, v) for k, v in sorted(kv.items()))
    info_bytes = bytearray()
    offset = 0
    infos = []
    for name, dtype, shape, raw in tensors:
        # tensor offsets are relative to the aligned tensor-data section
        infos.append((name, dtype, shape, offset, len(raw)))
        offset += len(raw)
        offset = (offset + alignment - 1) // alignment * alignment
    for name, dtype, shape, off, _size in infos:
        info_bytes += _gguf_string(name)
        info_bytes += struct.pack("<I", len(shape))
        info_bytes += struct.pack("<" + "Q" * len(shape), *shape)
        info_bytes += struct.pack("<I", dtype)
        info_bytes += struct.pack("<Q", off)
    header = _GGUF_MAGIC + struct.pack("<IQQ", _GGUF_VERSION, len(tensors), len(kv))
    prefix_len = len(header) + len(kv_bytes) + len(info_bytes)
    data_start = (prefix_len + alignment - 1) // alignment * alignment
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        fh.write(header)
        fh.write(kv_bytes)
        fh.write(info_bytes)
        fh.write(b"\0" * (data_start - prefix_len))
        cursor = 0
        for _name, _dtype, _shape, raw in tensors:
            current = data_start + cursor
            aligned = (current + alignment - 1) // alignment * alignment
            if aligned > current:
                fh.write(b"\0" * (aligned - current))
                cursor += aligned - current
            fh.write(raw)
            cursor += len(raw)
            pad = (alignment - (cursor % alignment)) % alignment
            if pad:
                fh.write(b"\0" * pad)
                cursor += pad
    return out


def convert_native(*, fmt: str, model: Any, output: str | Path, **kwargs: Any) -> Path:
    fmt = fmt.lower()
    if fmt == "gguf":
        state = model.state_dict() if hasattr(model, "state_dict") else model
        if not isinstance(state, Mapping):
            raise TypeError("GGUF conversion expects a model or state-dict mapping")
        return export_gguf(state, output, **kwargs)
    require_native(fmt)
    raise NotImplementedError(
        f"native {fmt} conversion requires the installed runtime's model-specific API; "
        "the adapter intentionally refuses an unsafe generic conversion"
    )


__all__ = ["FormatCapability", "NativeFormatUnavailable", "capabilities", "require_native", "export_gguf", "convert_native"]
