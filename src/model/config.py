"""Validation and zero-allocation sizing for GPT model configurations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


_ALIASES = {
    "hidden_size": ("dim", "model_dim"),
    "layers": ("num_layers", "num_hidden_layers"),
    "heads": ("num_attention_heads",),
    "kv_heads": ("num_kv_heads", "num_key_value_heads"),
    "max_position": ("context_length", "max_position_embeddings"),
    "ffn_hidden_size": ("intermediate_size",),
}


def normalize_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy using the engine's legacy keys while accepting common aliases.

    Existing configurations remain valid. Aliases make future large-model specs
    easier to import, and conflicting duplicate values fail loudly.
    """
    normalized = dict(config)
    for canonical, aliases in _ALIASES.items():
        candidates = [(canonical, normalized[canonical])] if canonical in normalized else []
        candidates.extend((name, normalized[name]) for name in aliases if name in normalized)
        if not candidates:
            continue
        value = candidates[0][1]
        conflicts = [name for name, candidate in candidates[1:] if candidate != value]
        if conflicts:
            names = ", ".join([candidates[0][0], *conflicts])
            raise ValueError(f"conflicting model configuration values for {names}")
        normalized[canonical] = value
    return normalized


@dataclass(frozen=True)
class ModelSize:
    parameters: int
    active_parameters_per_token: int
    parameter_bytes_fp32: int
    parameter_bytes_bf16: int
    kv_cache_bytes_bf16_per_sequence: int


def estimate_model_size(config: Mapping[str, Any]) -> ModelSize:
    """Calculate parameter and inference KV-cache sizes without building a model."""
    cfg = normalize_model_config(config)
    validate_moe_config(cfg)
    required = ("vocab_size", "hidden_size", "layers", "heads", "max_position")
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"missing required model configuration keys: {', '.join(missing)}")

    vocab = _positive_int(cfg["vocab_size"], "vocab_size")
    dim = _positive_int(cfg["hidden_size"], "hidden_size")
    layers = _positive_int(cfg["layers"], "layers")
    heads = _positive_int(cfg["heads"], "heads")
    kv_heads = _positive_int(cfg.get("kv_heads", heads), "kv_heads")
    context = _positive_int(cfg["max_position"], "max_position")
    if dim % heads:
        raise ValueError("hidden_size must be divisible by heads")
    if heads % kv_heads:
        raise ValueError("heads must be divisible by kv_heads")
    head_dim = dim // heads
    if str(cfg.get("position_type", "learned")).lower() == "rotary" and head_dim % 2:
        raise ValueError("attention head dimension must be even when using rotary positions")

    multiple = _positive_int(cfg.get("ffn_multiple_of", 1), "ffn_multiple_of")
    requested_ffn = cfg.get("ffn_hidden_size")
    if requested_ffn is None:
        expansion = float(cfg.get("ffn_expansion_factor", 4.0))
        if not math.isfinite(expansion) or expansion <= 0:
            raise ValueError("ffn_expansion_factor must be finite and positive")
        requested_ffn = math.ceil(dim * expansion)
    ffn = math.ceil(_positive_int(requested_ffn, "ffn_hidden_size") / multiple) * multiple

    attention_bias = bool(cfg.get("attention_bias", True))
    ffn_bias = bool(cfg.get("ffn_bias", True))
    norm_bias = bool(cfg.get("norm_bias", True))
    gated = str(cfg.get("ffn_activation", "gelu")).lower() in {"swiglu", "geglu"}
    kv_dim = kv_heads * head_dim

    parameters = vocab * dim
    if str(cfg.get("position_type", "learned")).lower() == "learned":
        parameters += context * dim
    attention = dim * (dim + 2 * kv_dim) + dim * dim
    if attention_bias:
        attention += dim + 2 * kv_dim + dim
    feed_forward = dim * ffn * (2 if gated else 1) + ffn * dim
    if ffn_bias:
        feed_forward += ffn * (2 if gated else 1) + dim
    norm = 2 * dim * (2 if norm_bias else 1)
    ffn_type = str(cfg.get("ffn_type", "dense")).lower()
    experts = _positive_int(cfg.get("num_experts", 1), "num_experts") if ffn_type == "moe" else 1
    active_experts = (_positive_int(cfg.get("experts_per_token", 1), "experts_per_token")
                      if ffn_type == "moe" else 1)
    router = dim * experts + (experts if bool(cfg.get("router_bias", False)) else 0)
    total_per_layer = attention + feed_forward * experts + norm + (router if ffn_type == "moe" else 0)
    active_per_layer = attention + feed_forward * active_experts + norm + (router if ffn_type == "moe" else 0)
    parameters += layers * total_per_layer
    parameters += dim * (2 if norm_bias else 1)
    if not bool(cfg.get("tie_word_embeddings", True)):
        parameters += vocab * dim
    if bool(cfg.get("lm_head_bias", False)):
        parameters += vocab

    kv_cache_elements = 2 * layers * kv_heads * context * head_dim
    # Embeddings, final norm, and head are active for every token.
    active_parameters = parameters - layers * (total_per_layer - active_per_layer)
    return ModelSize(parameters, active_parameters, parameters * 4, parameters * 2,
                     kv_cache_elements * 2)


def validate_moe_config(config: Mapping[str, Any]) -> None:
    """Validate sparse-FFN controls without allocating model weights."""
    ffn_type = str(config.get("ffn_type", "dense")).lower()
    if ffn_type not in {"dense", "moe"}:
        raise ValueError("ffn_type must be 'dense' or 'moe'")
    if ffn_type == "dense":
        _validate_attention_backend(config)
        return
    experts = _positive_int(config.get("num_experts", 1), "num_experts")
    active = _positive_int(config.get("experts_per_token", 1), "experts_per_token")
    if active > experts:
        raise ValueError("experts_per_token cannot exceed num_experts")
    jitter = float(config.get("router_jitter", 0.0))
    if not math.isfinite(jitter) or jitter < 0:
        raise ValueError("router_jitter must be finite and non-negative")
    _validate_attention_backend(config)


def _validate_attention_backend(config: Mapping[str, Any]) -> None:
    backend = str(config.get("attention_backend", "auto")).lower()
    if backend not in {"auto", "sdpa", "eager"}:
        raise ValueError("attention_backend must be auto, sdpa, or eager")


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value
