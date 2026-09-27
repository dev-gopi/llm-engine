"""Native quantized-format capability and export contracts.

GGUF is implemented as a real GGUF v3 binary container. GPTQ/AWQ remain
provider-specific because their packed-kernel layouts differ by runtime.
"""

from __future__ import annotations

import importlib.util
import struct
from dataclasses import dataclass
from pathlib import Path
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
            out[fmt] = FormatCapability(
                fmt, True, None, "built-in GGUF v3 writer available"
            )
            continue
        ok = importlib.util.find_spec(module) is not None
        out[fmt] = FormatCapability(
            fmt, ok, module, "available" if ok else f"{label} is not installed"
        )
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
_KV_INT32 = 5
_KV_FLOAT32 = 6
_KV_BOOL = 7
_KV_ARRAY = 9


def _gguf_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _array_value(values: list[Any] | tuple[Any, ...]) -> bytes:
    if not values:
        raise ValueError("GGUF arrays must not be empty")
    first = values[0]
    if all(isinstance(value, str) for value in values):
        element_type = _KV_STRING
        payload = b"".join(_gguf_string(value) for value in values)
    elif all(isinstance(value, bool) for value in values):
        element_type = _KV_BOOL
        payload = b"".join(struct.pack("<?", value) for value in values)
    elif all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        element_type = _KV_INT32
        payload = b"".join(struct.pack("<i", value) for value in values)
    elif all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
        element_type = _KV_FLOAT32
        payload = b"".join(struct.pack("<f", float(value)) for value in values)
    else:
        raise TypeError(f"unsupported GGUF array value type: {type(first).__name__}")
    return struct.pack("<IQ", element_type, len(values)) + payload


def _kv(
    key: str, value: str | int | float | bool | list[Any] | tuple[Any, ...]
) -> bytes:
    out = _gguf_string(key)
    if isinstance(value, bool):
        return out + struct.pack("<I?", _KV_BOOL, value)
    if isinstance(value, int):
        return out + struct.pack("<II", _KV_UINT32, value)
    if isinstance(value, float):
        return out + struct.pack("<If", _KV_FLOAT32, value)
    if isinstance(value, (list, tuple)):
        return out + struct.pack("<I", _KV_ARRAY) + _array_value(value)
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
    raise ValueError(
        f"unsupported GGUF tensor dtype: {value.dtype}; convert to float16/float32 first"
    )


def export_gguf(
    state_dict: Mapping[str, Any],
    output: str | Path,
    *,
    architecture: str = "llm-engine",
    model_name: str = "llm-engine",
    metadata: Mapping[str, Any] | None = None,
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



def _llama_rope_permute(weight: Any, heads: int) -> Any:
    """Convert Q/K projection rows to llama.cpp's RoPE tensor layout."""
    if heads < 1 or weight.shape[0] % (heads * 2):
        raise ValueError("attention projection shape is incompatible with LLaMA RoPE")
    return (
        weight.reshape(heads, 2, weight.shape[0] // heads // 2, *weight.shape[1:])
        .swapaxes(1, 2)
        .reshape(weight.shape)
    )


def minigpt_llama_state_dict(model: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    """Map a LLaMA-compatible MiniGPT checkpoint to llama.cpp tensor names."""
    import torch

    required = {
        "position_type": "rotary",
        "norm_type": "rms_norm",
        "ffn_activation": "swiglu",
        "ffn_type": "dense",
        "attention_pattern": "dense",
    }
    for key, expected in required.items():
        actual = str(config.get(key, expected)).lower()
        if actual != expected:
            raise ValueError(
                f"llama.cpp GGUF export requires {key}={expected!r}; got {actual!r}"
            )
    boolean_requirements = {
        "pre_norm": True,
        "attention_bias": False,
        "ffn_bias": False,
        "norm_bias": False,
        "lm_head_bias": False,
        "causal_attention": True,
        "qk_norm": False,
    }
    for key, expected in boolean_requirements.items():
        actual = bool(config.get(key, expected))
        if actual is not expected:
            raise ValueError(
                f"llama.cpp GGUF export requires {key}={expected}; got {actual}"
            )
    if str(config.get("rope_scaling_type", "none")).lower() != "none":
        raise ValueError("llama.cpp GGUF export currently requires rope_scaling_type='none'")
    if int(config.get("mtp_num_predictions", 0) or 0):
        raise ValueError("llama.cpp GGUF export does not support MTP heads")

    state = model.state_dict() if hasattr(model, "state_dict") else model
    if not isinstance(state, Mapping):
        raise TypeError("MiniGPT llama.cpp export expects a model or state dict")
    layers = int(config["layers"])
    heads = int(config["heads"])
    kv_heads = int(config.get("kv_heads") or heads)
    out: dict[str, Any] = {
        "token_embd.weight": state["tok.embedding.weight"],
        "output_norm.weight": state["norm.weight"],
        "output.weight": state["head.weight"],
    }
    for index in range(layers):
        src = f"blocks.{index}."
        dst = f"blk.{index}."
        q_name = src + "attn.q_proj.weight"
        if q_name in state:
            query = state[q_name]
            key = state[src + "attn.k_proj.weight"]
            value = state[src + "attn.v_proj.weight"]
        else:
            fused = state[src + "attn.qkv_proj.weight"]
            dim = int(config["hidden_size"])
            if fused.shape[0] != 3 * dim:
                raise ValueError("fused QKV projection is incompatible with LLaMA export")
            query, key, value = torch.split(fused, dim, dim=0)
        out[dst + "attn_q.weight"] = _llama_rope_permute(query, heads)
        out[dst + "attn_k.weight"] = _llama_rope_permute(key, kv_heads)
        out[dst + "attn_v.weight"] = value
        out[dst + "attn_output.weight"] = state[src + "attn.out_proj.weight"]
        out[dst + "attn_norm.weight"] = state[src + "attention_norm.weight"]
        out[dst + "ffn_norm.weight"] = state[src + "ffn_norm.weight"]
        fused_ffn = state[src + "ffn.in_proj.weight"]
        if fused_ffn.shape[0] % 2:
            raise ValueError("SwiGLU input projection must have an even output dimension")
        gate, up = fused_ffn.chunk(2, dim=0)
        out[dst + "ffn_gate.weight"] = gate
        out[dst + "ffn_up.weight"] = up
        out[dst + "ffn_down.weight"] = state[src + "ffn.out_proj.weight"]
    return out


def llama_cpp_metadata(config: Mapping[str, Any], tokenizer: Any) -> dict[str, Any]:
    """Build llama.cpp GGUF metadata for this engine's byte-level BPE tokenizer."""
    if getattr(tokenizer, "tokenizer_type", None) != "byte_level_bpe":
        raise ValueError("llama.cpp MiniGPT export currently requires byte_level_bpe")
    heads = int(config["heads"])
    hidden = int(config["hidden_size"])
    kv_heads = int(config.get("kv_heads") or heads)
    tokens = [tokenizer.id_to_token[index] for index in range(tokenizer.vocab_size)]
    special_ids = set(tokenizer.special_tokens.values())
    token_types = [3 if index in special_ids else 1 for index in range(tokenizer.vocab_size)]
    metadata: dict[str, Any] = {
        "llama.context_length": int(config["max_position"]),
        "llama.embedding_length": hidden,
        "llama.block_count": int(config["layers"]),
        "llama.feed_forward_length": int(config["ffn_hidden_size"]),
        "llama.rope.dimension_count": hidden // heads,
        "llama.rope.freq_base": float(config.get("rope_base", 10000.0)),
        "llama.attention.head_count": heads,
        "llama.attention.head_count_kv": kv_heads,
        "llama.attention.layer_norm_rms_epsilon": float(config.get("norm_eps", 1e-5)),
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "gpt-2",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.token_type": token_types,
        "tokenizer.ggml.add_bos_token": False,
        "tokenizer.ggml.add_eos_token": False,
    }
    merges = [f"{left} {right}" for left, right in tokenizer.bpe.merges]
    if merges:
        metadata["tokenizer.ggml.merges"] = merges
    token_keys = {
        "<|bos|>": "tokenizer.ggml.bos_token_id",
        "<|eos|>": "tokenizer.ggml.eos_token_id",
        "<|unk|>": "tokenizer.ggml.unknown_token_id",
        "<|pad|>": "tokenizer.ggml.padding_token_id",
    }
    for token, key in token_keys.items():
        if token in tokenizer.special_tokens:
            metadata[key] = int(tokenizer.special_tokens[token])
    return metadata


def export_llama_cpp_gguf(
    model: Any,
    config: Mapping[str, Any],
    tokenizer: Any,
    output: str | Path,
    *,
    model_name: str = "gopi",
) -> Path:
    """Export the LLaMA-compatible MiniGPT subset as a llama.cpp GGUF v3 file."""
    state = minigpt_llama_state_dict(model, config)
    return export_gguf(
        state,
        output,
        architecture="llama",
        model_name=model_name,
        metadata=llama_cpp_metadata(config, tokenizer),
    )

def convert_native(*, fmt: str, model: Any, output: str | Path, **kwargs: Any) -> Path:
    fmt = fmt.lower()
    if fmt == "gguf":
        state = model.state_dict() if hasattr(model, "state_dict") else model
        if not isinstance(state, Mapping):
            raise TypeError("GGUF conversion expects a model or state-dict mapping")
        return export_gguf(state, output, **kwargs)
    converter = kwargs.pop("converter", None)
    if converter is not None:
        result = converter(model=model, output=Path(output), format=fmt, **kwargs)
        return Path(result or output)
    require_native(fmt)
    if fmt == "gptq":
        # AutoGPTQ requires a HuggingFace-compatible model/tokenizer and calibration
        # examples.  When the caller supplies such an object, use its native API.
        quantize = getattr(model, "quantize", None)
        save = getattr(model, "save_quantized", None)
        examples = kwargs.pop("examples", None)
        if callable(quantize) and callable(save):
            if examples is None:
                raise ValueError("GPTQ conversion requires calibration examples")
            quantize(examples, **kwargs)
            save(str(output))
            return Path(output)
    if fmt == "awq":
        quantize = getattr(model, "quantize", None)
        save = getattr(model, "save_quantized", None)
        tokenizer = kwargs.pop("tokenizer", None)
        quant_config = kwargs.pop("quant_config", None)
        if callable(quantize) and callable(save):
            if tokenizer is None:
                raise ValueError("AWQ conversion requires a tokenizer")
            quantize(tokenizer, quant_config=quant_config or kwargs)
            save(str(output))
            return Path(output)
    raise NativeFormatUnavailable(
        f"installed {fmt} runtime requires a model-specific quantizable wrapper; "
        "supply a native model or an explicit converter callable"
    )


__all__ = [
    "FormatCapability",
    "NativeFormatUnavailable",
    "capabilities",
    "require_native",
    "export_gguf",
    "export_llama_cpp_gguf",
    "llama_cpp_metadata",
    "minigpt_llama_state_dict",
    "convert_native",
]
