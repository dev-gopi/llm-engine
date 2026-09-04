"""Causal multi-head self-attention with Grouped Query Attention (GQA) and RoPE support."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .positional import apply_rotary_pos_emb
from .kv_cache import StaticLayerKVCache

KeyValueCache: TypeAlias = tuple[Tensor, Tensor] | StaticLayerKVCache
AttentionOutput: TypeAlias = Tensor | tuple[Tensor, KeyValueCache]


class MultiHeadAttention(nn.Module):
    """GPT-style self-attention with fused/GQA QKV projection, RoPE support, and KV caching.

    Boolean masks use ``True`` for positions that may be attended to. Floating
    masks are additive attention biases, normally zero for allowed positions and
    ``-inf`` for blocked positions. Cached keys and values use shape
    ``[batch, kv_heads, sequence, head_dim]``.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        *,
        kv_heads: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        causal: bool = True,
        qk_norm: bool = False,
        qk_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self._validate_configuration(dim, heads, kv_heads, dropout, initializer_range)
        self.dim = dim
        self.heads = heads
        self.kv_heads = kv_heads if kv_heads is not None else heads
        self.num_kv_groups = heads // self.kv_heads
        self.head_dim = dim // heads
        self.dropout = float(dropout)
        self.causal = bool(causal)
        if not math.isfinite(qk_norm_eps) or qk_norm_eps <= 0:
            raise ValueError("qk_norm_eps must be finite and positive")
        self.qk_norm = bool(qk_norm)
        self.qk_norm_eps = float(qk_norm_eps)
        self.initializer_range = float(initializer_range)

        factory_kwargs = {"device": device, "dtype": dtype}
        if self.kv_heads == self.heads:
            self.qkv_proj = nn.Linear(dim, 3 * dim, bias=bias, **factory_kwargs)
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
        else:
            self.qkv_proj = None
            self.q_proj = nn.Linear(dim, dim, bias=bias, **factory_kwargs)
            self.k_proj = nn.Linear(dim, self.kv_heads * self.head_dim, bias=bias, **factory_kwargs)
            self.v_proj = nn.Linear(dim, self.kv_heads * self.head_dim, bias=bias, **factory_kwargs)

        self.out_proj = nn.Linear(dim, dim, bias=bias, **factory_kwargs)
        self._mask_cache: dict[tuple[Any, ...], Tensor] = {}
        self.tensor_parallel_group = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.qkv_proj is not None:
            nn.init.normal_(self.qkv_proj.weight, mean=0.0, std=self.initializer_range)
            if self.qkv_proj.bias is not None:
                nn.init.zeros_(self.qkv_proj.bias)
        else:
            for proj in (self.q_proj, self.k_proj, self.v_proj):
                if proj is not None:
                    nn.init.normal_(proj.weight, mean=0.0, std=self.initializer_range)
                    if proj.bias is not None:
                        nn.init.zeros_(proj.bias)

        nn.init.normal_(self.out_proj.weight, mean=0.0, std=self.initializer_range)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def project_qkv(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Project to Q [batch, heads, seq, head_dim], K & V [batch, kv_heads, seq, head_dim]."""
        self._validate_hidden_states(hidden_states)
        batch_size, sequence_length, _ = hidden_states.shape

        if self.qkv_proj is not None:
            projected = self.qkv_proj(hidden_states)
            projected = projected.view(batch_size, sequence_length, 3, self.heads, self.head_dim)
            query, key, value = projected.permute(2, 0, 3, 1, 4).unbind(0)
        else:
            assert self.q_proj is not None and self.k_proj is not None and self.v_proj is not None
            query = self.q_proj(hidden_states).view(batch_size, sequence_length, self.heads, self.head_dim).transpose(1, 2)
            key = self.k_proj(hidden_states).view(batch_size, sequence_length, self.kv_heads, self.head_dim).transpose(1, 2)
            value = self.v_proj(hidden_states).view(batch_size, sequence_length, self.kv_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            scale = math.sqrt(self.head_dim)
            query = F.normalize(query.float(), dim=-1, eps=self.qk_norm_eps).to(query.dtype) * scale
            key = F.normalize(key.float(), dim=-1, eps=self.qk_norm_eps).to(key.dtype) * scale
        return query, key, value

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        *,
        rotary_pos_emb: tuple[Tensor, Tensor] | None = None,
        position_ids: Tensor | None = None,
        past_key_value: KeyValueCache | None = None,
        use_cache: bool = False,
        is_causal: bool | None = None,
    ) -> AttentionOutput:
        query, key, value = self.project_qkv(hidden_states)

        if rotary_pos_emb is not None:
            cos, sin = rotary_pos_emb
            query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids=position_ids)

        batch_size, _, query_length, _ = query.shape
        past_length = 0

        if isinstance(past_key_value, StaticLayerKVCache):
            past_length = past_key_value.length
            key, value = past_key_value.append(key, value)
        elif past_key_value is not None:
            past_key, past_value = self._validate_cache(past_key_value, batch_size, key)
            past_length = past_key.size(2)
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)

        present_key_value = past_key_value if use_cache and isinstance(
            past_key_value, StaticLayerKVCache
        ) else ((key, value) if use_cache else None)
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
                enable_gqa=self.num_kv_groups > 1,
            )
        else:  # pragma: no cover
            key_attn = key.repeat_interleave(self.num_kv_groups, dim=1)
            value_attn = value.repeat_interleave(self.num_kv_groups, dim=1)
            attended = self._attention_fallback(
                query,
                key_attn,
                value_attn,
                prepared_mask,
                dropout_probability,
                kernel_is_causal,
            )

        output_width = self.heads * self.head_dim
        output = attended.transpose(1, 2).contiguous().view(batch_size, query_length, output_width)
        output = self.out_proj(output)
        if self.tensor_parallel_group is not None:
            torch.distributed.all_reduce(output, group=self.tensor_parallel_group)
        if use_cache:
            assert present_key_value is not None
            return output, present_key_value
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
            cache_key = (query_length, key_length, past_length, str(device))
            if cache_key in self._mask_cache:
                causal_mask = self._mask_cache[cache_key]
            else:
                query_positions = torch.arange(query_length, device=device) + past_length
                key_positions = torch.arange(key_length, device=device)
                causal_mask = (key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)).view(
                    1, 1, query_length, key_length
                )
                if len(self._mask_cache) < 32:
                    self._mask_cache[cache_key] = causal_mask

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
        expected_prefix = (batch_size, self.kv_heads)
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
            kv_heads=(int(config["kv_heads"]) if config.get("kv_heads") is not None else None),
            dropout=float(config.get("attention_dropout", 0.0)),
            bias=bool(config.get("attention_bias", True)),
            causal=bool(config.get("causal_attention", True)),
            qk_norm=bool(config.get("qk_norm", False)),
            qk_norm_eps=float(config.get("qk_norm_eps", 1e-6)),
            initializer_range=float(config.get("initializer_range", 0.02)),
            device=device,
            dtype=dtype,
        )

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, heads={self.heads}, kv_heads={self.kv_heads}, "
            f"head_dim={self.head_dim}, dropout={self.dropout}, causal={self.causal}"
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
        dim: int, heads: int, kv_heads: int | None, dropout: float, initializer_range: float
    ) -> None:
        for name, value in (("dim", dim), ("heads", heads)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        if kv_heads is not None:
            if not isinstance(kv_heads, int) or isinstance(kv_heads, bool) or kv_heads < 1:
                raise ValueError("kv_heads must be a positive integer")
            if heads % kv_heads != 0:
                raise ValueError("heads must be divisible by kv_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        if not math.isfinite(initializer_range) or initializer_range <= 0:
            raise ValueError("initializer_range must be finite and positive")
