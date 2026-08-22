"""AdamW construction with correct decay parameter groups."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn
from torch.optim import AdamW


def build_adamw(
    model: nn.Module,
    *,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    fused: bool | None = None,
) -> AdamW:
    """Exclude biases and normalization/vector parameters from weight decay."""
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if parameter.ndim < 2 or name.endswith("bias") else decay).append(parameter)
    kwargs: dict[str, Any] = {"lr": learning_rate, "betas": betas, "eps": eps}
    if fused is not None:
        kwargs["fused"] = fused
    return AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        **kwargs,
    )


def adamw_from_config(model: nn.Module, config: Mapping[str, Any]) -> AdamW:
    return build_adamw(
        model,
        learning_rate=float(config.get("learning_rate", 3e-4)),
        weight_decay=float(config.get("weight_decay", 0.01)),
        betas=(float(config.get("beta1", 0.9)), float(config.get("beta2", 0.95))),
        eps=float(config.get("adam_epsilon", 1e-8)),
        fused=config.get("fused_optimizer"),
    )


__all__ = ["AdamW", "adamw_from_config", "build_adamw"]
