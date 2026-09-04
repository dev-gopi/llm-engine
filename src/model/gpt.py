"""Production GPT-style decoder-only language model assembly with GQA, RoPE, and checkpointing."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch import Tensor

from utils.logger import get_logger

from .attention import KeyValueCache
from .kv_cache import StaticLayerKVCache
from .config import normalize_model_config
from .embedding import TokenEmbedding
from .layer_norm import build_normalization
from .positional import PositionalEmbedding, RotaryPositionalEmbedding, SinusoidalPositionalEmbedding
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
        kv_heads: int | None = None,
        position_type: str = "learned",
        position_initializer_range: float | None = None,
        rope_base: float = 10000.0,
        rope_scale: float = 1.0,
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
        qk_norm: bool = False,
        qk_norm_eps: float = 1e-6,
        ffn_hidden_dim: int | None = None,
        ffn_expansion_factor: float = 4.0,
        ffn_multiple_of: int = 1,
        ffn_activation: str = "gelu",
        ffn_dropout: float = 0.0,
        ffn_bias: bool = True,
        gradient_checkpointing: bool = False,
        logit_softcap: float | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self._validate_configuration(vocab_size, dim, layers, heads, max_pos, embedding_dropout)
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_positions = max_pos
        self.position_type = str(position_type).lower()
        self.tie_word_embeddings = bool(tie_word_embeddings)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        if logit_softcap is not None and (not math.isfinite(logit_softcap) or logit_softcap <= 0):
            raise ValueError("logit_softcap must be finite and positive when provided")
        self.logit_softcap = float(logit_softcap) if logit_softcap is not None else None

        if self.position_type not in {"learned", "rotary", "sinusoidal", "none"}:
            raise ValueError(f"Unsupported position_type: {position_type!r}")

        self.tok = TokenEmbedding(
            vocab_size,
            dim,
            padding_idx=padding_idx,
            initializer_range=initializer_range,
            scale_embeddings=scale_embeddings,
            freeze=freeze_embeddings,
            device=device,
            dtype=dtype,
        )

        if self.position_type == "learned":
            self.pos: nn.Module | None = PositionalEmbedding(
                max_pos,
                dim,
                initializer_range=(position_initializer_range or initializer_range),
                device=device,
                dtype=dtype,
            )
            self.rotary_emb: RotaryPositionalEmbedding | None = None
        elif self.position_type == "rotary":
            self.pos = None
            head_dim = dim // heads
            self.rotary_emb = RotaryPositionalEmbedding(
                head_dim,
                max_position_embeddings=max_pos,
                base=rope_base,
                scaling_factor=rope_scale,
                device=device,
                dtype=dtype,
            )
        elif self.position_type == "sinusoidal":
            self.pos = SinusoidalPositionalEmbedding(
                max_pos, dim, device=device, dtype=dtype
            )
            self.rotary_emb = None
        else:
            self.pos = None
            self.rotary_emb = None

        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(
                dim,
                heads,
                kv_heads=kv_heads,
                norm_type=norm_type,
                norm_eps=norm_eps,
                norm_bias=norm_bias,
                pre_norm=pre_norm,
                residual_dropout=residual_dropout,
                residual_scale=residual_scale,
                attention_dropout=attention_dropout,
                attention_bias=attention_bias,
                ffn_hidden_dim=ffn_hidden_dim,
                causal_attention=causal_attention,
                qk_norm=qk_norm,
                qk_norm_eps=qk_norm_eps,
                ffn_expansion_factor=ffn_expansion_factor,
                ffn_multiple_of=ffn_multiple_of,
                ffn_activation=ffn_activation,
                ffn_dropout=ffn_dropout,
                ffn_bias=ffn_bias,
                initializer_range=initializer_range,
                device=device,
                dtype=dtype,
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

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

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
        self._validate_inputs(token_ids)
        if not isinstance(position_offset, int) or isinstance(position_offset, bool):
            raise TypeError("position_offset must be an integer")
        if position_offset < 0:
            raise ValueError("position_offset must be non-negative")
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("past_key_values must contain one cache per Transformer block")
        cached_length = 0
        if past_key_values:
            for cache in past_key_values:
                if isinstance(cache, StaticLayerKVCache):
                    continue
                if not isinstance(cache, tuple) or len(cache) != 2:
                    raise TypeError("each past_key_values entry must be a (key, value) tuple")
                if any(not isinstance(tensor, Tensor) or tensor.ndim != 4 for tensor in cache):
                    raise ValueError("cached keys and values must have four dimensions")
            cached_length = (
                past_key_values[0].length
                if isinstance(past_key_values[0], StaticLayerKVCache)
                else past_key_values[0][0].shape[2]
            )
            if any(
                (cache.length if isinstance(cache, StaticLayerKVCache) else cache[0].shape[2])
                != cached_length
                or (
                    not isinstance(cache, StaticLayerKVCache)
                    and cache[1].shape[2] != cached_length
                )
                for cache in past_key_values
            ):
                raise ValueError("all cached keys and values must have the same sequence length")
            if position_offset == 0:
                position_offset = cached_length

        seq_len = token_ids.shape[1]
        self._validate_attention_mask(
            attention_mask,
            batch_size=token_ids.shape[0],
            sequence_length=seq_len,
            cached_length=cached_length,
        )
        current_attention_mask = attention_mask
        if attention_mask is not None and attention_mask.shape[1] != seq_len:
            current_attention_mask = attention_mask[:, -seq_len:]
        block_attention_mask = attention_mask
        if attention_mask is not None and cached_length and attention_mask.shape[1] == seq_len:
            prefix_mask = torch.ones(
                (token_ids.shape[0], cached_length),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            block_attention_mask = torch.cat((prefix_mask, attention_mask), dim=1)
        rotary_pos_emb: tuple[Tensor, Tensor] | None = None

        if self.position_type == "learned" and self.pos is not None:
            learned_position_ids = position_ids
            if (
                learned_position_ids is None
                and attention_mask is not None
                and attention_mask.shape[1] != seq_len
            ):
                learned_position_ids = (
                    attention_mask.long().cumsum(dim=1).sub(1).clamp_min(0)[:, -seq_len:]
                )
            positions = self.pos(
                token_ids,
                position_ids=learned_position_ids,
                position_offset=position_offset,
                attention_mask=current_attention_mask,
            )
            hidden_states = self.embedding_dropout(self.tok(token_ids) + positions)
        elif self.position_type == "sinusoidal" and self.pos is not None:
            positions = self.pos(token_ids, position_offset=position_offset)
            hidden_states = self.embedding_dropout(self.tok(token_ids) + positions)
        elif self.position_type == "rotary" and self.rotary_emb is not None:
            hidden_states = self.embedding_dropout(self.tok(token_ids))
            rope_length = position_offset + seq_len
            if position_ids is not None:
                position_ids = self._validate_position_ids(
                    position_ids, token_ids.shape[0], seq_len, token_ids.device
                )
                if not torch.compiler.is_compiling() and position_ids.numel():
                    rope_length = max(rope_length, int(position_ids.max()) + 1)
            rotary_pos_emb = self.rotary_emb(hidden_states, seq_len=rope_length)
            # The RoPE cache covers the full prefix, while an incremental
            # decoding call only projects the newly supplied tokens. Slice to
            # those positions unless explicit position IDs will index it.
            if position_ids is None:
                start = position_offset
                stop = position_offset + seq_len
                rotary_pos_emb = (
                    rotary_pos_emb[0][:, :, start:stop, :],
                    rotary_pos_emb[1][:, :, start:stop, :],
                )
        else:
            hidden_states = self.embedding_dropout(self.tok(token_ids))

        present_key_values: list[KeyValueCache] = []
        for index, block in enumerate(self.blocks):
            past_kv = past_key_values[index] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training and not use_cache:
                def create_custom_forward(target_block: TransformerBlock):
                    def custom_forward(*inputs: Any) -> Tensor:
                        out = target_block(
                            inputs[0],
                            attention_mask=inputs[1],
                            rotary_pos_emb=rotary_pos_emb,
                            position_ids=position_ids,
                        )
                        assert isinstance(out, Tensor)
                        return out

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    block_attention_mask,
                    use_reentrant=False,
                )
            else:
                block_output = block(
                    hidden_states,
                    attention_mask=block_attention_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    position_ids=position_ids,
                    past_key_value=past_kv,
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
        if self.logit_softcap is not None:
            logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return (logits, tuple(present_key_values)) if use_cache else logits

    def tie_weights(self) -> None:
        """Share token embedding and vocabulary projection weights."""
        self.tok.tie_weights(self.head)

    def resize_token_embeddings(
        self,
        new_vocab_size: int,
        *,
        pad_to_multiple_of: int | None = None,
        init_strategy: str = "normal",
    ) -> int:
        """Resize input/output vocabulary matrices and restore weight tying."""
        effective_size = self.tok.resize(
            new_vocab_size, pad_to_multiple_of=pad_to_multiple_of, init_strategy=init_strategy
        )
        old_head = self.head
        replacement = nn.Linear(
            self.dim,
            effective_size,
            bias=old_head.bias is not None,
            device=old_head.weight.device,
            dtype=old_head.weight.dtype,
        )
        rows = min(old_head.out_features, effective_size)
        with torch.no_grad():
            if init_strategy == "zero":
                replacement.weight.zero_()
            elif init_strategy == "mean":
                mean_vec = old_head.weight.mean(dim=0)
                replacement.weight.copy_(mean_vec.unsqueeze(0).expand(effective_size, -1))
            else:
                nn.init.normal_(replacement.weight, mean=0.0, std=self.tok.initializer_range)

            replacement.weight[:rows].copy_(old_head.weight[:rows])
            if replacement.bias is not None:
                replacement.bias.zero_()
                replacement.bias[:rows].copy_(old_head.bias[:rows])

        self.head = replacement
        self.vocab_size = effective_size
        if self.tie_word_embeddings:
            self.tie_weights()
        logger.info("Resized GPT vocabulary to %d tokens (init_strategy=%s)", effective_size, init_strategy)
        return effective_size

    def num_parameters(self, *, trainable_only: bool = False) -> int:
        parameters = self.parameters()
        if trainable_only:
            parameters = (parameter for parameter in parameters if parameter.requires_grad)
        return sum(parameter.numel() for parameter in parameters)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "MiniGPT":
        """Build a complete model from model configuration values."""
        config = normalize_model_config(config)
        pos_type = str(config.get("position_type", "learned")).lower()
        valid_pos_types = {"learned", "rotary", "sinusoidal", "none"}
        if pos_type not in valid_pos_types:
            raise ValueError(f"position_type must be one of {sorted(valid_pos_types)}, got {pos_type!r}")

        return cls(
            vocab_size=int(config["vocab_size"]),
            dim=int(config["hidden_size"]),
            layers=int(config["layers"]),
            heads=int(config["heads"]),
            kv_heads=(int(config["kv_heads"]) if config.get("kv_heads") is not None else None),
            max_pos=int(config["max_position"]),
            position_type=pos_type,
            rope_base=float(config.get("rope_base", 10000.0)),
            rope_scale=float(config.get("rope_scale", 1.0)),
            initializer_range=float(config.get("initializer_range", 0.02)),
            position_initializer_range=float(
                config.get("position_initializer_range", config.get("initializer_range", 0.02))
            ),
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
            qk_norm=bool(config.get("qk_norm", False)),
            qk_norm_eps=float(config.get("qk_norm_eps", 1e-6)),
            ffn_hidden_dim=(
                int(config["ffn_hidden_size"])
                if config.get("ffn_hidden_size") is not None
                else None
            ),
            ffn_expansion_factor=float(config.get("ffn_expansion_factor", 4.0)),
            ffn_multiple_of=int(config.get("ffn_multiple_of", 1)),
            ffn_activation=str(config.get("ffn_activation", "gelu")),
            ffn_dropout=float(config.get("ffn_dropout", 0.0)),
            ffn_bias=bool(config.get("ffn_bias", True)),
            gradient_checkpointing=bool(config.get("gradient_checkpointing", False)),
            logit_softcap=(
                float(config["logit_softcap"])
                if config.get("logit_softcap") is not None else None
            ),
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _validate_configuration(
        vocab_size: int,
        dim: int,
        layers: int,
        heads: int,
        max_pos: int,
        embedding_dropout: float,
    ) -> None:
        for name, value in (
            ("vocab_size", vocab_size),
            ("dim", dim),
            ("layers", layers),
            ("heads", heads),
            ("max_pos", max_pos),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        if not math.isfinite(embedding_dropout) or not 0 <= embedding_dropout < 1:
            raise ValueError("embedding_dropout must satisfy 0 <= dropout < 1")

    @staticmethod
    def _validate_inputs(token_ids: Tensor) -> None:
        if not isinstance(token_ids, Tensor) or token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("token_ids must use an integer dtype")
        if token_ids.shape[1] == 0:
            raise ValueError("token_ids sequence cannot be empty")

    @staticmethod
    def _validate_attention_mask(
        attention_mask: Tensor | None,
        *,
        batch_size: int,
        sequence_length: int,
        cached_length: int,
    ) -> None:
        if attention_mask is None:
            return
        if not isinstance(attention_mask, Tensor) or attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [batch, sequence]")
        valid_shapes = {(batch_size, sequence_length)}
        if cached_length:
            valid_shapes.add((batch_size, cached_length + sequence_length))
        if tuple(attention_mask.shape) not in valid_shapes:
            expected = " or ".join(str(shape) for shape in sorted(valid_shapes))
            raise ValueError(f"attention_mask must have shape {expected}")
        if not torch.compiler.is_compiling() and not torch.all(
            (attention_mask == 0) | (attention_mask == 1)
        ):
            raise ValueError("attention_mask values must be binary")

    @staticmethod
    def _validate_position_ids(
        position_ids: Tensor, batch_size: int, sequence_length: int, device: torch.device
    ) -> Tensor:
        if not isinstance(position_ids, Tensor):
            raise TypeError("position_ids must be a torch.Tensor")
        if position_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("position_ids must use an integer dtype")
        if position_ids.ndim == 1 and position_ids.shape[0] == sequence_length:
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        elif position_ids.shape != (batch_size, sequence_length):
            raise ValueError("position_ids must have shape [sequence] or [batch, sequence]")
        if position_ids.device != device:
            raise ValueError("position_ids and token_ids must be on the same device")
        if not torch.compiler.is_compiling() and position_ids.numel() and int(position_ids.min()) < 0:
            raise IndexError("position IDs must be non-negative")
        return position_ids.to(dtype=torch.long)
