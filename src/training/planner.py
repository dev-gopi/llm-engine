"""Zero-allocation training compute, memory, runtime, and cost estimates."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from model.config import estimate_model_size, normalize_model_config


def optimizer_steps_for_epochs(
    batches_per_epoch: int, epochs: int, accumulation_steps: int
) -> int:
    """Count optimizer updates when partial accumulation is flushed per epoch."""
    if min(batches_per_epoch, epochs, accumulation_steps) < 1:
        raise ValueError("batches_per_epoch, epochs, and accumulation_steps must be positive")
    return epochs * math.ceil(batches_per_epoch / accumulation_steps)


@dataclass(frozen=True)
class TrainingPlan:
    parameters: int
    training_tokens: int
    tokens_per_optimizer_step: int
    optimizer_steps: int
    training_flops: float
    model_state_gib_per_gpu: float
    activation_gib_per_gpu: float
    estimated_peak_gib_per_gpu: float
    estimated_hours: float | None
    estimated_cost: float | None
    fits_memory: bool | None
    assumptions: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_training(
    model_config: Mapping[str, Any], training_config: Mapping[str, Any], *,
    training_tokens: int, gpus: int = 1, hardware_tflops: float | None = None,
    utilization: float = 0.35, gpu_memory_gib: float | None = None,
    hourly_cost_per_gpu: float | None = None,
) -> TrainingPlan:
    if training_tokens < 1 or gpus < 1:
        raise ValueError("training_tokens and gpus must be positive")
    if hardware_tflops is not None and hardware_tflops <= 0:
        raise ValueError("hardware_tflops must be positive")
    if not 0 < utilization <= 1:
        raise ValueError("utilization must be in (0, 1]")
    if gpu_memory_gib is not None and gpu_memory_gib <= 0:
        raise ValueError("gpu_memory_gib must be positive")
    if hourly_cost_per_gpu is not None and hourly_cost_per_gpu < 0:
        raise ValueError("hourly_cost_per_gpu cannot be negative")

    model = normalize_model_config(model_config)
    size = estimate_model_size(model)
    batch = int(training_config.get("batch_size", 1))
    sequence = int(training_config.get("max_sequence_length", model["max_position"]))
    accumulation = int(training_config.get("gradient_accumulation_steps", 1))
    if min(batch, sequence, accumulation) < 1:
        raise ValueError("batch, sequence length, and accumulation must be positive")
    tokens_per_step = batch * sequence * accumulation * gpus
    steps = math.ceil(training_tokens / tokens_per_step)
    flops = float(6 * size.parameters * training_tokens)

    strategy = str(training_config.get("distributed_strategy", "ddp"))
    sharding = gpus if strategy.startswith("fsdp") else 1
    # The current trainer keeps FP32 parameters and AdamW uses two FP32 moments.
    state_bytes = size.parameters * (4 + 4 + 8) / sharding
    if not strategy.startswith("fsdp") and training_config.get("ema_decay") is not None:
        state_bytes += size.parameters * 4
    hidden = int(model["hidden_size"])
    layers = int(model["layers"])
    activation_factor = 4 if bool(model.get("gradient_checkpointing", False)) else 12
    activation_bytes = batch * sequence * hidden * layers * activation_factor * 4
    # Reserve 20% for temporary kernels, allocator fragmentation, and communication buffers.
    peak_bytes = 1.2 * (state_bytes + activation_bytes)
    gib = 1024**3
    peak_gib = peak_bytes / gib

    hours = None
    if hardware_tflops is not None:
        effective_flops_per_second = hardware_tflops * 1e12 * utilization * gpus
        hours = flops / effective_flops_per_second / 3600
    cost = hours * hourly_cost_per_gpu * gpus if hours is not None and hourly_cost_per_gpu is not None else None
    fits = peak_gib <= gpu_memory_gib if gpu_memory_gib is not None else None
    return TrainingPlan(
        parameters=size.parameters,
        training_tokens=training_tokens,
        tokens_per_optimizer_step=tokens_per_step,
        optimizer_steps=steps,
        training_flops=flops,
        model_state_gib_per_gpu=state_bytes / gib,
        activation_gib_per_gpu=activation_bytes / gib,
        estimated_peak_gib_per_gpu=peak_gib,
        estimated_hours=hours,
        estimated_cost=cost,
        fits_memory=fits,
        assumptions={
            "dense_training_flops_per_token": "6 * parameters",
            "optimizer": "AdamW with FP32 parameters, gradients, and moments",
            "activation_factor": activation_factor,
            "memory_safety_factor": 1.2,
            "distributed_strategy": strategy,
            "utilization": utilization,
            "estimate_only": True,
        },
    )
