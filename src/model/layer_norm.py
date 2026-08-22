"""Normalization layers used by the Transformer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LayerNorm(nn.Module):
    """Layer normalization with optional bias and strict shape validation."""

    def __init__(
        self,
        dim: int,
        *,
        eps: float = 1e-5,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        _validate_configuration(dim, eps)
        self.dim = dim
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.bias = (
            nn.Parameter(torch.zeros(dim, device=device, dtype=dtype)) if bias else None
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        _validate_hidden_states(hidden_states, self.dim)
        return F.layer_norm(
            hidden_states,
            normalized_shape=(self.dim,),
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
        )

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}, bias={self.bias is not None}"


class RMSNorm(nn.Module):
    """Root-mean-square normalization with float32 accumulation for low precision."""

    def __init__(
        self,
        dim: int,
        *,
        eps: float = 1e-5,
        bias: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        _validate_configuration(dim, eps)
        self.dim = dim
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.bias = (
            nn.Parameter(torch.zeros(dim, device=device, dtype=dtype)) if bias else None
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        _validate_hidden_states(hidden_states, self.dim)
        accumulation_dtype = (
            torch.float32
            if hidden_states.dtype in (torch.float16, torch.bfloat16)
            else hidden_states.dtype
        )
        variance = hidden_states.to(accumulation_dtype).square().mean(dim=-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.eps).to(hidden_states.dtype)
        output = normalized * self.weight
        if self.bias is not None:
            output = output + self.bias
        return output

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}, bias={self.bias is not None}"


def build_normalization(
    norm_type: str,
    dim: int,
    *,
    eps: float = 1e-5,
    bias: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> LayerNorm | RMSNorm:
    normalized_name = norm_type.lower()
    if normalized_name in {"layernorm", "layer_norm"}:
        return LayerNorm(dim, eps=eps, bias=bias, device=device, dtype=dtype)
    if normalized_name in {"rmsnorm", "rms_norm"}:
        return RMSNorm(dim, eps=eps, bias=bias, device=device, dtype=dtype)
    raise ValueError("norm_type must be 'layer_norm' or 'rms_norm'")


def normalization_from_config(
    config: Mapping[str, Any],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> LayerNorm | RMSNorm:
    return build_normalization(
        str(config.get("norm_type", "layer_norm")),
        int(config["hidden_size"]),
        eps=float(config.get("norm_eps", 1e-5)),
        bias=bool(config.get("norm_bias", True)),
        device=device,
        dtype=dtype,
    )


def _validate_configuration(dim: int, eps: float) -> None:
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise TypeError("dim must be an integer")
    if dim < 1:
        raise ValueError("dim must be positive")
    if not isinstance(eps, (int, float)) or isinstance(eps, bool):
        raise TypeError("eps must be a number")
    if not 0 < eps < 1:
        raise ValueError("eps must satisfy 0 < eps < 1")


def _validate_hidden_states(hidden_states: Tensor, dim: int) -> None:
    if not isinstance(hidden_states, Tensor):
        raise TypeError("hidden_states must be a torch.Tensor")
    if hidden_states.ndim < 2:
        raise ValueError("hidden_states must have at least two dimensions")
    if hidden_states.shape[-1] != dim:
        raise ValueError(
            f"hidden_states final dimension must be {dim}, got {hidden_states.shape[-1]}"
        )
    if not hidden_states.is_floating_point():
        raise TypeError("hidden_states must use a floating-point dtype")
