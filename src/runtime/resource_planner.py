"""Hardware-neutral model memory and deployment planning.

The planner intentionally separates *storage arithmetic* from *quality claims*.
It can estimate whether a configuration fits a memory budget, but it never
claims that an aggressive precision (for example Q1_0) preserves model quality
without benchmark evidence for the concrete checkpoint.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from model.config import estimate_model_size, normalize_model_config

WEIGHT_BITS: dict[str, float] = {
    "fp32": 32.0,
    "float32": 32.0,
    "fp16": 16.0,
    "float16": 16.0,
    "bf16": 16.0,
    "bfloat16": 16.0,
    "int8": 8.0,
    "int4": 4.0,
    # llama.cpp Q1_0 uses one sign bit plus a 16-bit scale per 128 weights.
    "q1_0": 1.125,
}

KV_BITS: dict[str, float] = {
    "fp32": 32.0,
    "fp16": 16.0,
    "bf16": 16.0,
    "int8": 8.0,
    "int4": 4.0,
}


@dataclass(frozen=True)
class RuntimeMemoryEstimate:
    weight_precision: str
    kv_precision: str
    context_length: int
    batch_size: int
    parameter_count: int
    active_parameters_per_token: int
    weight_bytes: int
    kv_cache_bytes: int
    linear_state_bytes: int
    runtime_state_bytes: int
    estimated_total_bytes: int
    estimated_total_gib: float
    memory_margin: float
    fits_budget: bool | None
    budget_bytes: int | None
    notes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _precision_bits(name: str, table: Mapping[str, float], kind: str) -> tuple[str, float]:
    normalized = str(name).lower()
    if normalized not in table:
        choices = ", ".join(sorted(table))
        raise ValueError(f"unsupported {kind} precision {name!r}; choose one of {choices}")
    return normalized, float(table[normalized])


def estimate_inference_memory(
    config: Mapping[str, Any],
    *,
    context_length: int | None = None,
    batch_size: int = 1,
    weight_precision: str = "bf16",
    kv_precision: str = "bf16",
    memory_budget_bytes: int | None = None,
    memory_margin: float = 0.10,
) -> RuntimeMemoryEstimate:
    """Estimate resident model + cache/state bytes for inference.

    ``estimate_model_size`` reports BF16 cache/state at the configured maximum
    context.  This function rescales the full-attention KV component to the
    requested context and precision. Linear-attention recurrent state is fixed
    with context length and scales only with state precision and batch size.
    The result excludes allocator fragmentation and backend-specific workspaces;
    ``memory_margin`` reserves capacity for those costs.
    """
    cfg = normalize_model_config(config)
    model_size = estimate_model_size(cfg)
    configured_context = int(cfg["max_position"])
    context = configured_context if context_length is None else int(context_length)
    if context < 1:
        raise ValueError("context_length must be positive")
    if context > configured_context:
        raise ValueError("context_length cannot exceed model max_position")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not math.isfinite(memory_margin) or not 0 <= memory_margin < 1:
        raise ValueError("memory_margin must satisfy 0 <= memory_margin < 1")
    if memory_budget_bytes is not None and memory_budget_bytes < 1:
        raise ValueError("memory_budget_bytes must be positive when provided")

    weight_name, weight_bits = _precision_bits(weight_precision, WEIGHT_BITS, "weight")
    kv_name, kv_bits = _precision_bits(kv_precision, KV_BITS, "KV-cache")
    weight_bytes = math.ceil(model_size.parameters * weight_bits / 8.0)

    # ModelSize stores BF16 (16-bit) values for one configured-length sequence.
    context_ratio = context / configured_context
    kv_cache_bytes = math.ceil(
        model_size.kv_cache_bytes_bf16_per_sequence
        * context_ratio
        * (kv_bits / 16.0)
        * batch_size
    )
    linear_state_bytes = math.ceil(
        model_size.linear_state_bytes_bf16_per_sequence
        * (kv_bits / 16.0)
        * batch_size
    )
    runtime_state_bytes = kv_cache_bytes + linear_state_bytes
    subtotal = weight_bytes + runtime_state_bytes
    estimated_total = math.ceil(subtotal / (1.0 - memory_margin)) if memory_margin < 1 else subtotal
    fits = None if memory_budget_bytes is None else estimated_total <= memory_budget_bytes

    notes = [
        "memory is an analytical estimate, not a measured peak",
        "backend workspaces, CUDA graphs, fragmentation, and multimodal towers are not explicitly modeled",
    ]
    if weight_name == "q1_0":
        notes.append("Q1_0 is a deployment-size estimate; checkpoint quality requires quantization-aware validation")
    if kv_name in {"int8", "int4"}:
        notes.append("low-bit KV memory assumes a backend that stores cache natively at the requested precision")
    if model_size.linear_attention_layers:
        notes.append("linear-attention recurrent state is fixed-size with context length")

    return RuntimeMemoryEstimate(
        weight_precision=weight_name,
        kv_precision=kv_name,
        context_length=context,
        batch_size=batch_size,
        parameter_count=model_size.parameters,
        active_parameters_per_token=model_size.active_parameters_per_token,
        weight_bytes=weight_bytes,
        kv_cache_bytes=kv_cache_bytes,
        linear_state_bytes=linear_state_bytes,
        runtime_state_bytes=runtime_state_bytes,
        estimated_total_bytes=estimated_total,
        estimated_total_gib=estimated_total / (1024 ** 3),
        memory_margin=memory_margin,
        fits_budget=fits,
        budget_bytes=memory_budget_bytes,
        notes=tuple(notes),
    )


def deployment_matrix(
    config: Mapping[str, Any],
    *,
    context_length: int | None = None,
    batch_size: int = 1,
    memory_budget_bytes: int | None = None,
    weight_precisions: tuple[str, ...] = ("bf16", "int8", "int4", "q1_0"),
    kv_precisions: tuple[str, ...] = ("bf16", "int8", "int4"),
) -> list[dict[str, Any]]:
    """Return deterministic footprint alternatives ordered by total memory."""
    rows = [
        estimate_inference_memory(
            config,
            context_length=context_length,
            batch_size=batch_size,
            weight_precision=weights,
            kv_precision=kv,
            memory_budget_bytes=memory_budget_bytes,
        ).as_dict()
        for weights in weight_precisions
        for kv in kv_precisions
    ]
    rows.sort(key=lambda item: (item["estimated_total_bytes"], item["weight_precision"], item["kv_precision"]))
    return rows
