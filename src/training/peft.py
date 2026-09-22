"""Parameter-efficient fine-tuning utilities for the native MiniGPT model."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

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


def lora_adapter_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Return a portable adapter-only state dictionary."""
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if ".lora_a" in name or ".lora_b" in name
    }


def load_lora_adapter(model: nn.Module, state: Mapping[str, Tensor]) -> None:
    """Atomically replace only LoRA weights on an already-adapted model."""
    expected = lora_adapter_state_dict(model)
    if set(state) != set(expected):
        raise ValueError("adapter state does not match the model's LoRA modules")
    for name, value in state.items():
        if not isinstance(value, Tensor) or value.shape != expected[name].shape:
            raise ValueError(f"adapter parameter has incompatible shape: {name}")
    with torch.no_grad():
        named_parameters = dict(model.named_parameters())
        for name, value in state.items():
            named_parameters[name].copy_(value.to(named_parameters[name]))


def merge_and_unload(model: nn.Module) -> nn.Module:
    """Fold LoRA residuals into fresh ``nn.Linear`` modules and remove adapters."""
    replacements: list[tuple[nn.Module, str, LoRALinear]] = []
    for full_name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        parent_name, _, child_name = full_name.rpartition(".")
        replacements.append((model.get_submodule(parent_name) if parent_name else model, child_name, module))
    with torch.no_grad():
        for parent, child_name, module in replacements:
            base = module.base
            merged = nn.Linear(
                base.in_features, base.out_features, bias=base.bias is not None,
                device=base.weight.device, dtype=base.weight.dtype,
            )
            merged.weight.copy_(base.weight + (module.lora_b @ module.lora_a).to(base.weight) * module.scaling)
            if base.bias is not None:
                merged.bias.copy_(base.bias)
            setattr(parent, child_name, merged)
    return model


def _targets(value: Any) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError("PEFT target_modules must be a sequence of module names")
    targets = tuple(str(item).strip() for item in value)
    if not targets or any(not item for item in targets):
        raise ValueError("PEFT target_modules must not be empty")
    return targets


__all__ = [
    "DEFAULT_LORA_TARGETS", "LoRALinear", "apply_lora", "has_lora",
    "load_lora_adapter", "lora_adapter_state_dict", "merge_and_unload",
]


def prepare_qlora(model: nn.Module, config: Mapping[str, Any]) -> dict[str, Any]:
    """Prepare a low-bit-compatible frozen base followed by LoRA adapters.

    The engine keeps the base weights in their already-loaded dtype; external
    GPTQ/AWQ backends can supply the low-bit module. This function enforces the
    QLoRA contract and records the quantization provenance used for the run.
    """
    quantization = str(config.get("quantization", "int4")).lower()
    if quantization not in {"int4", "int8"}:
        raise ValueError("QLoRA quantization must be int4 or int8")
    details = apply_lora(model, config.get("lora", config))
    details["profile"] = "qlora"
    details["quantization"] = quantization
    details["base_model_frozen"] = all(not p.requires_grad for name,p in model.named_parameters() if "lora_" not in name)
    return details
