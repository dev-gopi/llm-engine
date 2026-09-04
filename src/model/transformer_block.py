"""A configurable pre-norm / post-norm Transformer block with residual connections and GQA/RoPE support."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from .attention import KeyValueCache, MultiHeadAttention
from .feed_forward import FeedForward
from .layer_norm import build_normalization


class TransformerBlock(nn.Module):
    """Compose self-attention and FFN sublayers with normalized residual paths."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        *,
        kv_heads: int | None = None,
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        norm_bias: bool = True,
        pre_norm: bool = True,
        residual_dropout: float = 0.0,
        residual_scale: float = 1.0,
        attention_dropout: float = 0.0,
        attention_bias: bool = True,
        causal_attention: bool = True,
        qk_norm: bool = False,
        qk_norm_eps: float = 1e-6,
        ffn_hidden_dim: int | None = None,
        ffn_expansion_factor: float = 4.0,
        ffn_multiple_of: int = 1,
        ffn_activation: str = "gelu",
        ffn_dropout: float = 0.0,
        ffn_bias: bool = True,
        initializer_range: float = 0.02,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= residual_dropout < 1.0:
            raise ValueError("residual_dropout must satisfy 0 <= dropout < 1")
        if not math.isfinite(residual_scale) or residual_scale < 0:
            raise ValueError("residual_scale must be finite and non-negative")

        self.pre_norm = bool(pre_norm)
        self.residual_scale = float(residual_scale)
        self.attn = MultiHeadAttention(
            dim,
            heads,
            kv_heads=kv_heads,
            dropout=attention_dropout,
            bias=attention_bias,
            causal=causal_attention,
            qk_norm=qk_norm,
            qk_norm_eps=qk_norm_eps,
            initializer_range=initializer_range,
            device=device,
            dtype=dtype,
        )
        self.ffn = FeedForward(
            dim,
            hidden_dim=ffn_hidden_dim,
            expansion_factor=ffn_expansion_factor,
            multiple_of=ffn_multiple_of,
            activation=ffn_activation,
            dropout=ffn_dropout,
            bias=ffn_bias,
            initializer_range=initializer_range,
            device=device,
            dtype=dtype,
        )
        self.attention_norm = build_normalization(
            norm_type, dim, eps=norm_eps, bias=norm_bias, device=device, dtype=dtype
        )
        self.ffn_norm = build_normalization(
            norm_type, dim, eps=norm_eps, bias=norm_bias, device=device, dtype=dtype
        )
        self.attention_residual_dropout = nn.Dropout(residual_dropout)
        self.ffn_residual_dropout = nn.Dropout(residual_dropout)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        *,
        rotary_pos_emb: tuple[Tensor, Tensor] | None = None,
        position_ids: Tensor | None = None,
        past_key_value: KeyValueCache | None = None,
        use_cache: bool = False,
    ) -> Tensor | tuple[Tensor, KeyValueCache]:
        if self.pre_norm:
            attention_result = self.attn(
                self.attention_norm(hidden_states),
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            attention_update, present = self._unpack_attention(attention_result, use_cache)
            hidden_states = self._add_residual(
                hidden_states, attention_update, self.attention_residual_dropout
            )
            ffn_update = self.ffn(self.ffn_norm(hidden_states))
            output = self._add_residual(hidden_states, ffn_update, self.ffn_residual_dropout)
            return (output, present) if present is not None else output

        attention_result = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        attention_update, present = self._unpack_attention(attention_result, use_cache)
        hidden_states = self.attention_norm(
            self._add_residual(
                hidden_states, attention_update, self.attention_residual_dropout
            )
        )
        ffn_update = self.ffn(hidden_states)
        output = self.ffn_norm(
            self._add_residual(hidden_states, ffn_update, self.ffn_residual_dropout)
        )
        return (output, present) if present is not None else output

    @staticmethod
    def _unpack_attention(
        result: Tensor | tuple[Tensor, KeyValueCache], use_cache: bool
    ) -> tuple[Tensor, KeyValueCache | None]:
        if use_cache:
            if not isinstance(result, tuple):
                raise RuntimeError("attention did not return a requested cache")
            return result
        if isinstance(result, tuple):
            raise RuntimeError("attention unexpectedly returned a cache")
        return result, None

    def _add_residual(
        self, residual: Tensor, update: Tensor, dropout: nn.Dropout
    ) -> Tensor:
        if residual.shape != update.shape:
            raise ValueError("residual and sublayer update shapes must match")
        return residual + dropout(update) * self.residual_scale

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "TransformerBlock":
        return cls(
            dim=int(config["hidden_size"]),
            heads=int(config["heads"]),
            kv_heads=(int(config["kv_heads"]) if config.get("kv_heads") is not None else None),
            norm_type=str(config.get("norm_type", "layer_norm")),
            norm_eps=float(config.get("norm_eps", 1e-5)),
            norm_bias=bool(config.get("norm_bias", True)),
            pre_norm=bool(config.get("pre_norm", True)),
            residual_dropout=float(config.get("residual_dropout", 0.0)),
            residual_scale=float(config.get("residual_scale", 1.0)),
            attention_dropout=float(config.get("attention_dropout", 0.0)),
            attention_bias=bool(config.get("attention_bias", True)),
            causal_attention=bool(config.get("causal_attention", True)),
            qk_norm=bool(config.get("qk_norm", False)),
            qk_norm_eps=float(config.get("qk_norm_eps", 1e-6)),
            ffn_hidden_dim=config.get("ffn_hidden_size"),
            ffn_expansion_factor=float(config.get("ffn_expansion_factor", 4.0)),
            ffn_multiple_of=int(config.get("ffn_multiple_of", 1)),
            ffn_activation=str(config.get("ffn_activation", "gelu")),
            ffn_dropout=float(config.get("ffn_dropout", 0.0)),
            ffn_bias=bool(config.get("ffn_bias", True)),
            initializer_range=float(config.get("initializer_range", 0.02)),
            device=device,
            dtype=dtype,
        )
