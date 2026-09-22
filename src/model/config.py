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
    linear_state_bytes_bf16_per_sequence: int = 0
    runtime_state_bytes_bf16_per_sequence: int = 0
    full_attention_layers: int = 0
    linear_attention_layers: int = 0


def resolve_attention_layer_pattern(config: Mapping[str, Any], layers: int | None = None) -> tuple[str, ...]:
    """Expand the configured attention pattern to one validated value per layer.

    ``attention_layer_pattern`` is a compact repeating cycle, for example
    ``[linear, linear, linear, dense]``.  This keeps legacy single-pattern
    configs fully compatible while allowing hybrid-attention research
    profiles without duplicating dozens of layer entries.
    """
    total_layers = layers if layers is not None else _positive_int(config["layers"], "layers")
    raw = config.get("attention_layer_pattern")
    if raw is None:
        raw = [str(config.get("attention_pattern", "dense")).lower()]
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("attention_layer_pattern must be a non-empty list or string")
    cycle = tuple(str(item).lower() for item in raw)
    supported = {"dense", "sliding_window", "linear"}
    unsupported = sorted(set(cycle) - supported)
    if unsupported:
        raise ValueError(f"unsupported attention pattern(s): {', '.join(unsupported)}")
    if "sliding_window" in cycle:
        window = config.get("attention_window")
        if not isinstance(window, int) or isinstance(window, bool) or window < 1:
            raise ValueError("attention_window must be a positive integer when sliding_window is used")
    return tuple(cycle[index % len(cycle)] for index in range(total_layers))


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

    layer_patterns = resolve_attention_layer_pattern(cfg, layers)
    linear_layers = sum(pattern == "linear" for pattern in layer_patterns)
    full_layers = layers - linear_layers
    kv_cache_elements = 2 * full_layers * kv_heads * context * head_dim
    # The reference linear backend stores one recurrent K vector and one K⊗V
    # matrix per query head. This state is context-length independent.
    linear_state_elements = linear_layers * heads * (head_dim + head_dim * head_dim)
    # Embeddings, final norm, and head are active for every token.
    active_parameters = parameters - layers * (total_per_layer - active_per_layer)
    kv_bytes = kv_cache_elements * 2
    linear_bytes = linear_state_elements * 2
    return ModelSize(
        parameters,
        active_parameters,
        parameters * 4,
        parameters * 2,
        kv_bytes,
        linear_bytes,
        kv_bytes + linear_bytes,
        full_layers,
        linear_layers,
    )


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
    capacity = config.get("moe_capacity_factor")
    if capacity is not None and (not math.isfinite(float(capacity)) or float(capacity) <= 0):
        raise ValueError("moe_capacity_factor must be finite and positive, or None")
    min_capacity = config.get("moe_min_capacity", 0)
    if not isinstance(min_capacity, int) or isinstance(min_capacity, bool) or min_capacity < 0:
        raise ValueError("moe_min_capacity must be a non-negative integer")
    _validate_attention_backend(config)


def _validate_attention_backend(config: Mapping[str, Any]) -> None:
    if "layers" in config:
        resolve_attention_layer_pattern(config, int(config["layers"]))
    else:
        pattern = str(config.get("attention_pattern", "dense")).lower()
        if pattern not in {"dense", "sliding_window", "linear"}:
            raise ValueError("attention_pattern must be dense, sliding_window, or linear")
        if pattern == "sliding_window" and (
            not isinstance(config.get("attention_window"), int)
            or isinstance(config.get("attention_window"), bool)
            or config["attention_window"] < 1
        ):
            raise ValueError("attention_window must be a positive integer for sliding_window attention")
    eps = float(config.get("linear_attention_eps", 1e-6))
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("linear_attention_eps must be finite and positive")
    chunk = config.get("linear_attention_chunk_size", 128)
    if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk < 1:
        raise ValueError("linear_attention_chunk_size must be a positive integer")
    backend = str(config.get("attention_backend", "auto")).lower()
    if backend not in {"auto", "sdpa", "eager"}:
        raise ValueError("attention_backend must be auto, sdpa, or eager")


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def context_scaling_profile(config: Mapping[str, Any], lengths=(512,1024,2048,4096,8192)) -> list[dict[str,int|str]]:
    """Return validated context variants without mutating the base model config."""
    cfg=normalize_model_config(config); base=int(cfg["max_position"])
    if any(int(x)<=0 for x in lengths): raise ValueError("context lengths must be positive")
    return [{"base_context":base,"context_length":int(x),"position_type":str(cfg.get("position_type","learned")),"rope_scaling_type":str(cfg.get("rope_scaling_type","none"))} for x in lengths]
