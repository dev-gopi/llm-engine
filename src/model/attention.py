"""Causal multi-head self-attention for decoder-only language models."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


KeyValueCache: TypeAlias = tuple[Tensor, Tensor]
AttentionOutput: TypeAlias = Tensor | tuple[Tensor, KeyValueCache]


class MultiHeadAttention(nn.Module):
    """GPT-style self-attention with fused QKV projection and KV caching.

    Boolean masks use ``True`` for positions that may be attended to. Floating
    masks are additive attention biases, normally zero for allowed positions and
    ``-inf`` for blocked positions. Cached keys and values use the shape
    ``[batch, heads, sequence, head_dim]``.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        *,
        dropout: float = 0.0,
        bias: bool = True,
        causal: bool = True,
        initializer_range: float = 0.02,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self._validate_configuration(dim, heads, dropout, initializer_range)
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = float(dropout)
        self.causal = bool(causal)
        self.initializer_range = float(initializer_range)

        factory_kwargs = {"device": device, "dtype": dtype}
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(dim, dim, bias=bias, **factory_kwargs)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.qkv_proj.weight, mean=0.0, std=self.initializer_range)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=self.initializer_range)
        if self.qkv_proj.bias is not None:
            nn.init.zeros_(self.qkv_proj.bias)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def project_qkv(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Project to Q, K, and V tensors shaped ``[batch, heads, seq, head_dim]``."""

        self._validate_hidden_states(hidden_states)
        batch_size, sequence_length, _ = hidden_states.shape
        projected = self.qkv_proj(hidden_states)
        projected = projected.view(batch_size, sequence_length, 3, self.heads, self.head_dim)
        query, key, value = projected.permute(2, 0, 3, 1, 4).unbind(0)
        return query, key, value

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        *,
        past_key_value: KeyValueCache | None = None,
        use_cache: bool = False,
        is_causal: bool | None = None,
    ) -> AttentionOutput:
        query, key, value = self.project_qkv(hidden_states)
        batch_size, _, query_length, _ = query.shape
        past_length = 0

        if past_key_value is not None:
            past_key, past_value = self._validate_cache(past_key_value, batch_size, key)
            past_length = past_key.size(2)
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)

        key_length = key.size(2)
        apply_causal = self.causal if is_causal is None else bool(is_causal)
        prepared_mask, kernel_is_causal = self._prepare_mask(
            attention_mask,
            batch_size=batch_size,
            query_length=query_length,
            key_length=key_length,
            past_length=past_length,
            causal=apply_causal,
            device=query.device,
            dtype=query.dtype,
        )

        dropout_probability = self.dropout if self.training else 0.0
        if hasattr(F, "scaled_dot_product_attention"):
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=prepared_mask,
                dropout_p=dropout_probability,
                is_causal=kernel_is_causal,
                scale=1.0 / math.sqrt(self.head_dim),
            )
        else:  # pragma: no cover - supported for older PyTorch deployments
            attended = self._attention_fallback(
                query,
                key,
                value,
                prepared_mask,
                dropout_probability,
                kernel_is_causal,
            )

        output = attended.transpose(1, 2).contiguous().view(batch_size, query_length, self.dim)
        output = self.out_proj(output)
        if use_cache:
            return output, (key, value)
        return output

    def _prepare_mask(
        self,
        attention_mask: Tensor | None,
        *,
        batch_size: int,
        query_length: int,
        key_length: int,
        past_length: int,
        causal: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor | None, bool]:
        # The native causal kernel is fastest when no cache/custom mask requires
        # constructing an explicit offset-aware mask.
        if attention_mask is None and causal and past_length == 0:
            return None, True
        if attention_mask is None and not causal:
            return None, False

        prepared = None
        if attention_mask is not None:
            prepared = self._normalize_mask(
                attention_mask,
                batch_size=batch_size,
                query_length=query_length,
                key_length=key_length,
                device=device,
                dtype=dtype,
            )

        if causal:
            query_positions = torch.arange(query_length, device=device) + past_length
            key_positions = torch.arange(key_length, device=device)
            causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            causal_mask = causal_mask.view(1, 1, query_length, key_length)
            if prepared is None:
                prepared = causal_mask
            elif prepared.dtype == torch.bool:
                prepared = prepared & causal_mask
            else:
                prepared = prepared.masked_fill(~causal_mask, float("-inf"))
        return prepared, False

    def _normalize_mask(
        self,
        mask: Tensor,
        *,
        batch_size: int,
        query_length: int,
        key_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if not isinstance(mask, Tensor):
            raise TypeError("attention_mask must be a torch.Tensor")
        if mask.ndim == 2:
            if tuple(mask.shape) != (batch_size, key_length):
                raise ValueError(
                    f"2D attention_mask must have shape {(batch_size, key_length)}, "
                    f"got {tuple(mask.shape)}"
                )
            mask = mask[:, None, None, :]
        elif mask.ndim == 3:
            if tuple(mask.shape) != (batch_size, query_length, key_length):
                raise ValueError(
                    "3D attention_mask must have shape "
                    f"{(batch_size, query_length, key_length)}, got {tuple(mask.shape)}"
                )
            mask = mask[:, None, :, :]
        elif mask.ndim == 4:
            if mask.shape[0] not in (1, batch_size):
                raise ValueError("4D attention_mask batch dimension is not broadcastable")
            if mask.shape[1] not in (1, self.heads):
                raise ValueError("4D attention_mask head dimension is not broadcastable")
            if mask.shape[2] not in (1, query_length) or mask.shape[3] != key_length:
                raise ValueError("4D attention_mask query/key dimensions are not broadcastable")
        else:
            raise ValueError("attention_mask must have 2, 3, or 4 dimensions")

        if mask.dtype == torch.bool:
            return mask.to(device=device)
        if not mask.is_floating_point():
            return mask.to(device=device, dtype=torch.bool)
        return mask.to(device=device, dtype=dtype)

    def _validate_cache(
        self,
        cache: KeyValueCache,
        batch_size: int,
        current_key: Tensor,
    ) -> KeyValueCache:
        if not isinstance(cache, tuple) or len(cache) != 2:
            raise TypeError("past_key_value must be a (key, value) tuple")
        key, value = cache
        expected_prefix = (batch_size, self.heads)
        for name, tensor in (("key", key), ("value", value)):
            if not isinstance(tensor, Tensor) or tensor.ndim != 4:
                raise ValueError(f"cached {name} must have four dimensions")
            if tensor.shape[:2] != expected_prefix or tensor.shape[3] != self.head_dim:
                raise ValueError(f"cached {name} has incompatible shape {tuple(tensor.shape)}")
            if tensor.device != current_key.device or tensor.dtype != current_key.dtype:
                raise ValueError(f"cached {name} must match the current tensor device and dtype")
        if key.shape != value.shape:
            raise ValueError("cached key and value shapes must match")
        return key, value

    def _attention_fallback(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None,
        dropout_probability: float,
        is_causal: bool,
    ) -> Tensor:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if is_causal:
            causal_mask = torch.ones(
                scores.shape[-2:], dtype=torch.bool, device=scores.device
            ).tril()
            scores = scores.masked_fill(~causal_mask, float("-inf"))
        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, float("-inf"))
            else:
                scores = scores + mask
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        probabilities = torch.nan_to_num(probabilities)
        probabilities = F.dropout(probabilities, p=dropout_probability, training=self.training)
        return torch.matmul(probabilities, value)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "MultiHeadAttention":
        return cls(
            dim=int(config["hidden_size"]),
            heads=int(config["heads"]),
            dropout=float(config.get("attention_dropout", 0.0)),
            bias=bool(config.get("attention_bias", True)),
            causal=bool(config.get("causal_attention", True)),
            initializer_range=float(config.get("initializer_range", 0.02)),
            device=device,
            dtype=dtype,
        )

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, heads={self.heads}, head_dim={self.head_dim}, "
            f"dropout={self.dropout}, causal={self.causal}"
        )

    def _validate_hidden_states(self, hidden_states: Tensor) -> None:
        if not isinstance(hidden_states, Tensor):
            raise TypeError("hidden_states must be a torch.Tensor")
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [batch, sequence, dim], "
                f"got {tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[-1] != self.dim:
            raise ValueError(
                f"hidden_states shape final dimension must be {self.dim}, "
                f"got {hidden_states.shape[-1]}"
            )
        if hidden_states.shape[1] == 0:
            raise ValueError("hidden_states sequence dimension cannot be empty")
        if not hidden_states.is_floating_point():
            raise TypeError("hidden_states must use a floating-point dtype")

    @staticmethod
    def _validate_configuration(
        dim: int, heads: int, dropout: float, initializer_range: float
    ) -> None:
        for name, value in (("dim", dim), ("heads", heads)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        if not math.isfinite(initializer_range) or initializer_range <= 0:
            raise ValueError("initializer_range must be finite and positive")
