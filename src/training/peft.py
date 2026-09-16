"""Parameter-efficient fine-tuning utilities for the native MiniGPT model."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "qkv_proj",
    "out_proj",
    "in_proj",
)


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank residual update."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("LoRA alpha must be finite and positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.Parameter(base.weight.new_empty(self.rank, base.in_features))
        self.lora_b = nn.Parameter(base.weight.new_zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, inputs: Tensor) -> Tensor:
        update = F.linear(F.linear(self.dropout(inputs), self.lora_a), self.lora_b)
        return self.base(inputs) + update * self.scaling


def apply_lora(model: nn.Module, config: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze ``model`` and replace selected linear projections with LoRA layers."""
    method = str(config.get("method", "lora")).lower()
    if method != "lora":
        raise ValueError(f"unsupported PEFT method: {method!r}")
    rank = int(config.get("rank", 8))
    alpha = float(config.get("alpha", 2 * rank))
    dropout = float(config.get("dropout", 0.0))
    targets = _targets(config.get("target_modules", DEFAULT_LORA_TARGETS))

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    replacements: list[tuple[nn.Module, str, nn.Linear, str]] = []
    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or isinstance(module, LoRALinear):
            continue
        if not any(full_name == target or full_name.endswith(f".{target}") for target in targets):
            continue
        parent_name, _, child_name = full_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        replacements.append((parent, child_name, module, full_name))
    if not replacements:
        raise ValueError(f"LoRA target_modules matched no linear layers: {list(targets)!r}")
    for parent, child_name, module, _ in replacements:
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "method": "lora",
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "target_modules": list(targets),
        "matched_modules": [name for *_, name in replacements],
        "trainable_parameters": trainable,
        "total_parameters": total,
    }


def has_lora(model: nn.Module) -> bool:
    return any(isinstance(module, LoRALinear) for module in model.modules())


def _targets(value: Any) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError("PEFT target_modules must be a sequence of module names")
    targets = tuple(str(item).strip() for item in value)
    if not targets or any(not item for item in targets):
        raise ValueError("PEFT target_modules must not be empty")
    return targets


__all__ = ["DEFAULT_LORA_TARGETS", "LoRALinear", "apply_lora", "has_lora"]
