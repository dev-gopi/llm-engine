"""Position-wise feed-forward network for Transformer blocks."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


STANDARD_ACTIVATIONS = frozenset({"gelu", "gelu_tanh", "relu", "silu"})
GATED_ACTIVATIONS = frozenset({"swiglu", "geglu"})
SUPPORTED_ACTIVATIONS = STANDARD_ACTIVATIONS | GATED_ACTIVATIONS


class FeedForward(nn.Module):
    """Apply an independent nonlinear transformation at every token position.

    Standard activations use ``Linear → activation → Linear``. SwiGLU and GEGLU
    use one fused input projection for the gate and value branches, followed by
    an output projection. The fused layout is efficient on accelerators and
    keeps checkpoint structure straightforward.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        *,
        expansion_factor: float = 4.0,
        multiple_of: int = 1,
        activation: str = "gelu",
        dropout: float = 0.0,
        bias: bool = True,
        initializer_range: float = 0.02,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self._validate_configuration(
            dim,
            hidden_dim,
            expansion_factor,
            multiple_of,
            activation,
            dropout,
            initializer_range,
        )
        requested_hidden_dim = hidden_dim or math.ceil(dim * expansion_factor)
        self.dim = dim
        self.hidden_dim = math.ceil(requested_hidden_dim / multiple_of) * multiple_of
        self.activation_name = activation
        self.dropout_probability = float(dropout)
        self.initializer_range = float(initializer_range)
        self.is_gated = activation in GATED_ACTIVATIONS

        factory_kwargs = {"device": device, "dtype": dtype}
        input_features = 2 * self.hidden_dim if self.is_gated else self.hidden_dim
        self.in_proj = nn.Linear(dim, input_features, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(self.hidden_dim, dim, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(self.dropout_probability)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.in_proj.weight, mean=0.0, std=self.initializer_range)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=self.initializer_range)
        if self.in_proj.bias is not None:
            nn.init.zeros_(self.in_proj.bias)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        self._validate_hidden_states(hidden_states)
        projected = self.in_proj(hidden_states)
        if self.is_gated:
            gate, value = projected.chunk(2, dim=-1)
            hidden = self._activate_gate(gate) * value
        else:
            hidden = self._activate(projected)
        return self.dropout(self.out_proj(hidden))

    def _activate(self, hidden_states: Tensor) -> Tensor:
        if self.activation_name == "gelu":
            return F.gelu(hidden_states)
        if self.activation_name == "gelu_tanh":
            return F.gelu(hidden_states, approximate="tanh")
        if self.activation_name == "relu":
            return F.relu(hidden_states)
        if self.activation_name == "silu":
            return F.silu(hidden_states)
        raise RuntimeError(f"unsupported standard activation: {self.activation_name}")

    def _activate_gate(self, gate: Tensor) -> Tensor:
        if self.activation_name == "swiglu":
            return F.silu(gate)
        if self.activation_name == "geglu":
            return F.gelu(gate)
        raise RuntimeError(f"unsupported gated activation: {self.activation_name}")

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "FeedForward":
        return cls(
            dim=int(config["hidden_size"]),
            hidden_dim=(
                int(config["ffn_hidden_size"])
                if config.get("ffn_hidden_size") is not None
                else None
            ),
            expansion_factor=float(config.get("ffn_expansion_factor", 4.0)),
            multiple_of=int(config.get("ffn_multiple_of", 1)),
            activation=str(config.get("ffn_activation", "gelu")),
            dropout=float(config.get("ffn_dropout", 0.0)),
            bias=bool(config.get("ffn_bias", True)),
            initializer_range=float(config.get("initializer_range", 0.02)),
            device=device,
            dtype=dtype,
        )

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, hidden_dim={self.hidden_dim}, "
            f"activation={self.activation_name!r}, dropout={self.dropout_probability}, "
            f"gated={self.is_gated}"
        )

    def _validate_hidden_states(self, hidden_states: Tensor) -> None:
        if not isinstance(hidden_states, Tensor):
            raise TypeError("hidden_states must be a torch.Tensor")
        if hidden_states.ndim < 2:
            raise ValueError("hidden_states must have at least two dimensions")
        if hidden_states.shape[-1] != self.dim:
            raise ValueError(
                f"hidden_states final dimension must be {self.dim}, "
                f"got {hidden_states.shape[-1]}"
            )
        if not hidden_states.is_floating_point():
            raise TypeError("hidden_states must use a floating-point dtype")

    @staticmethod
    def _validate_configuration(
        dim: int,
        hidden_dim: int | None,
        expansion_factor: float,
        multiple_of: int,
        activation: str,
        dropout: float,
        initializer_range: float,
    ) -> None:
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise TypeError("dim must be an integer")
        if dim < 1:
            raise ValueError("dim must be positive")
        if hidden_dim is not None:
            if not isinstance(hidden_dim, int) or isinstance(hidden_dim, bool):
                raise TypeError("hidden_dim must be an integer or None")
            if hidden_dim < 1:
                raise ValueError("hidden_dim must be positive")
        if not math.isfinite(expansion_factor) or expansion_factor <= 0:
            raise ValueError("expansion_factor must be finite and positive")
        if not isinstance(multiple_of, int) or isinstance(multiple_of, bool):
            raise TypeError("multiple_of must be an integer")
        if multiple_of < 1:
            raise ValueError("multiple_of must be positive")
        if activation not in SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {sorted(SUPPORTED_ACTIVATIONS)}, got {activation!r}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        if not math.isfinite(initializer_range) or initializer_range <= 0:
            raise ValueError("initializer_range must be finite and positive")
