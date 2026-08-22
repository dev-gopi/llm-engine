"""Production GPT-style decoder-only language model assembly."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from utils.logger import get_logger

from .embedding import TokenEmbedding
from .attention import KeyValueCache
from .layer_norm import build_normalization
from .positional import PositionalEmbedding
from .transformer_block import TransformerBlock

logger = get_logger(__name__)


class MiniGPT(nn.Module):
    """Compose embeddings, Transformer blocks, final norm, and LM head."""

    def __init__(
        self,
        vocab_size: int,
        dim: int = 128,
        layers: int = 4,
        heads: int = 4,
        max_pos: int = 512,
        initializer_range: float = 0.02,
        *,
        position_initializer_range: float | None = None,
        padding_idx: int | None = None,
        embedding_dropout: float = 0.0,
        scale_embeddings: bool = False,
        freeze_embeddings: bool = False,
        tie_word_embeddings: bool = True,
        lm_head_bias: bool = False,
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        norm_bias: bool = True,
        pre_norm: bool = True,
        residual_dropout: float = 0.0,
        residual_scale: float = 1.0,
        attention_dropout: float = 0.0,
        attention_bias: bool = True,
        causal_attention: bool = True,
        ffn_hidden_dim: int | None = None,
        ffn_expansion_factor: float = 4.0,
        ffn_multiple_of: int = 1,
        ffn_activation: str = "gelu",
        ffn_dropout: float = 0.0,
        ffn_bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self._validate_configuration(vocab_size, dim, layers, heads, max_pos, embedding_dropout)
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_positions = max_pos
        self.tie_word_embeddings = bool(tie_word_embeddings)

        self.tok = TokenEmbedding(
            vocab_size, dim, padding_idx=padding_idx,
            initializer_range=initializer_range, scale_embeddings=scale_embeddings,
            freeze=freeze_embeddings,
            device=device, dtype=dtype,
        )
        self.pos = PositionalEmbedding(
            max_pos, dim,
            initializer_range=(position_initializer_range or initializer_range),
            device=device, dtype=dtype
        )
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(
                dim, heads, norm_type=norm_type, norm_eps=norm_eps, norm_bias=norm_bias,
                pre_norm=pre_norm, residual_dropout=residual_dropout,
                residual_scale=residual_scale, attention_dropout=attention_dropout,
                attention_bias=attention_bias, ffn_hidden_dim=ffn_hidden_dim,
                causal_attention=causal_attention,
                ffn_expansion_factor=ffn_expansion_factor,
                ffn_multiple_of=ffn_multiple_of, ffn_activation=ffn_activation,
                ffn_dropout=ffn_dropout, ffn_bias=ffn_bias,
                initializer_range=initializer_range,
                device=device, dtype=dtype,
            )
            for _ in range(layers)
        )
        self.norm = build_normalization(
            norm_type, dim, eps=norm_eps, bias=norm_bias, device=device, dtype=dtype
        )
        self.head = nn.Linear(dim, vocab_size, bias=lm_head_bias, device=device, dtype=dtype)
        nn.init.normal_(self.head.weight, mean=0.0, std=initializer_range)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)
        if self.tie_word_embeddings:
            self.tie_weights()
        logger.debug("Initialized GPT with %d layers and %d parameters", layers, self.num_parameters())

    def forward(
        self,
        token_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        position_ids: Tensor | None = None,
        position_offset: int = 0,
        past_key_values: tuple[KeyValueCache, ...] | None = None,
        use_cache: bool = False,
    ) -> Tensor | tuple[Tensor, tuple[KeyValueCache, ...]]:
        self._validate_inputs(token_ids, attention_mask)
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("past_key_values must contain one cache per Transformer block")
        if past_key_values:
            cached_length = past_key_values[0][0].shape[2]
            if position_offset == 0:
                position_offset = cached_length
        positions = self.pos(
            token_ids, position_ids=position_ids, position_offset=position_offset,
            attention_mask=attention_mask,
        )
        hidden_states = self.embedding_dropout(self.tok(token_ids) + positions)
        present_key_values: list[KeyValueCache] = []
        for index, block in enumerate(self.blocks):
            block_output = block(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_values[index] if past_key_values is not None else None,
                use_cache=use_cache,
            )
            if use_cache:
                if not isinstance(block_output, tuple):
                    raise RuntimeError("Transformer block did not return a requested cache")
                hidden_states, present = block_output
                present_key_values.append(present)
            else:
                if isinstance(block_output, tuple):
                    raise RuntimeError("Transformer block unexpectedly returned a cache")
                hidden_states = block_output
        logits = self.head(self.norm(hidden_states))
        return (logits, tuple(present_key_values)) if use_cache else logits

    def tie_weights(self) -> None:
        """Share token embedding and vocabulary projection weights."""
        self.tok.tie_weights(self.head)

    def resize_token_embeddings(
        self, new_vocab_size: int, *, pad_to_multiple_of: int | None = None
    ) -> int:
        """Resize input/output vocabulary matrices and restore weight tying."""
        effective_size = self.tok.resize(new_vocab_size, pad_to_multiple_of=pad_to_multiple_of)
        old_head = self.head
        replacement = nn.Linear(
            self.dim, effective_size, bias=old_head.bias is not None,
            device=old_head.weight.device, dtype=old_head.weight.dtype,
        )
        with torch.no_grad():
            rows = min(old_head.out_features, effective_size)
            replacement.weight[:rows].copy_(old_head.weight[:rows])
            if replacement.bias is not None:
                replacement.bias.zero_()
                replacement.bias[:rows].copy_(old_head.bias[:rows])
        self.head = replacement
        self.vocab_size = effective_size
        if self.tie_word_embeddings:
            self.tie_weights()
        logger.info("Resized GPT vocabulary to %d tokens", effective_size)
        return effective_size

    def num_parameters(self, *, trainable_only: bool = False) -> int:
        parameters = self.parameters()
        if trainable_only:
            parameters = (parameter for parameter in parameters if parameter.requires_grad)
        return sum(parameter.numel() for parameter in parameters)

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], *,
        device: torch.device | str | None = None, dtype: torch.dtype | None = None,
    ) -> "MiniGPT":
        """Build a complete model from model configuration values."""
        if str(config.get("position_type", "learned")).lower() != "learned":
            raise ValueError("MiniGPT currently supports position_type='learned' only")
        return cls(
            vocab_size=int(config["vocab_size"]), dim=int(config["hidden_size"]),
            layers=int(config["layers"]), heads=int(config["heads"]),
            max_pos=int(config["max_position"]),
            initializer_range=float(config.get("initializer_range", 0.02)),
            position_initializer_range=float(config.get("position_initializer_range", config.get("initializer_range", 0.02))),
            padding_idx=config.get("padding_idx"),
            embedding_dropout=float(config.get("embedding_dropout", 0.0)),
            scale_embeddings=bool(config.get("scale_embeddings", False)),
            freeze_embeddings=bool(config.get("freeze_embeddings", False)),
            tie_word_embeddings=bool(config.get("tie_word_embeddings", True)),
            lm_head_bias=bool(config.get("lm_head_bias", False)),
            norm_type=str(config.get("norm_type", "layer_norm")),
            norm_eps=float(config.get("norm_eps", 1e-5)),
            norm_bias=bool(config.get("norm_bias", True)),
            pre_norm=bool(config.get("pre_norm", True)),
            residual_dropout=float(config.get("residual_dropout", 0.0)),
            residual_scale=float(config.get("residual_scale", 1.0)),
            attention_dropout=float(config.get("attention_dropout", 0.0)),
            attention_bias=bool(config.get("attention_bias", True)),
            causal_attention=bool(config.get("causal_attention", True)),
            ffn_hidden_dim=(int(config["ffn_hidden_size"]) if config.get("ffn_hidden_size") is not None else None),
            ffn_expansion_factor=float(config.get("ffn_expansion_factor", 4.0)),
            ffn_multiple_of=int(config.get("ffn_multiple_of", 1)),
            ffn_activation=str(config.get("ffn_activation", "gelu")),
            ffn_dropout=float(config.get("ffn_dropout", 0.0)),
            ffn_bias=bool(config.get("ffn_bias", True)), device=device, dtype=dtype,
        )

    @staticmethod
    def _validate_configuration(
        vocab_size: int, dim: int, layers: int, heads: int,
        max_pos: int, embedding_dropout: float,
    ) -> None:
        for name, value in (("vocab_size", vocab_size), ("dim", dim), ("layers", layers),
                            ("heads", heads), ("max_pos", max_pos)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        if not math.isfinite(embedding_dropout) or not 0 <= embedding_dropout < 1:
            raise ValueError("embedding_dropout must satisfy 0 <= dropout < 1")

    @staticmethod
    def _validate_inputs(token_ids: Tensor, attention_mask: Tensor | None) -> None:
        if not isinstance(token_ids, Tensor) or token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("token_ids must use an integer dtype")
        if token_ids.shape[1] == 0:
            raise ValueError("token_ids sequence cannot be empty")
        if attention_mask is not None and attention_mask.shape != token_ids.shape:
            raise ValueError("attention_mask must match token_ids shape")
