"""Execution engine for large sparse and hybrid Mixture of Experts models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from model.config import estimate_model_size
from model.gpt import MiniGPT
from utils.config import load_yaml
from utils.logger import get_logger

logger = get_logger(__name__)


def build_executable_moe_model(
    config: Mapping[str, Any] | str | Path,
    *,
    device: str | torch.device = "meta",
    dtype: torch.dtype = torch.bfloat16,
) -> MiniGPT:
    """Instantiate a sparse or hybrid MoE model ready for execution.

    Supports meta-device dry runs and multi-node sharding for 100B+ class
    architectures that exceed single-machine physical RAM.
    """
    if isinstance(config, (str, Path)):
        cfg = load_yaml(config)
    else:
        cfg = dict(config)

    model = MiniGPT.from_config(cfg, device=device, dtype=dtype, allow_planning=True)
    return model


def execute_moe_forward(
    model: MiniGPT,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute a forward pass through a sparse or hybrid MoE architecture."""
    return model(input_ids, attention_mask=attention_mask)


def execute_sparse_profile(
    config_path: str | Path,
    *,
    sequence_length: int = 16,
    batch_size: int = 1,
    device: str | torch.device = "meta",
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Any]:
    """Execute an end-to-end profile evaluation for a sparse/hybrid MoE configuration."""
    cfg = load_yaml(config_path)
    size = estimate_model_size(cfg)
    model = build_executable_moe_model(cfg, device=device, dtype=dtype)
    input_ids = torch.randint(
        0, model.vocab_size, (batch_size, sequence_length), device=device
    )
    with torch.inference_mode():
        logits = execute_moe_forward(model, input_ids)

    return {
        "architecture": "MiniGPT",
        "ffn_type": cfg.get("ffn_type", "dense"),
        "total_parameters": size.parameters,
        "active_parameters": size.active_parameters_per_token,
        "vocab_size": model.vocab_size,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "output_shape": list(logits.shape),
        "device": str(device),
        "executed": True,
    }


__all__ = [
    "build_executable_moe_model",
    "execute_moe_forward",
    "execute_sparse_profile",
]
