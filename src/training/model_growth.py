"""Shape-safe depth and append-only vocabulary growth for decoder models."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from model.gpt import MiniGPT


_VOCABULARY_PARAMETERS = frozenset({
    "tok.embedding.weight", "head.weight", "head.bias",
})


@dataclass(frozen=True)
class GrowthReport:
    source_layers: int
    target_layers: int
    source_vocab_size: int
    target_vocab_size: int
    copied_parameters: int
    new_parameters: int


def grow_model(
    source: MiniGPT,
    target: MiniGPT,
    *,
    embedding_init: str = "mean",
) -> GrowthReport:
    """Copy a smaller compatible model into a deeper/wider-vocabulary model.

    Existing blocks and vocabulary rows are copied exactly. Newly appended
    blocks are initialized as identity residual blocks by zeroing their
    attention and feed-forward output projections.
    """
    source_layers, target_layers = len(source.blocks), len(target.blocks)
    if target_layers <= source_layers:
        raise ValueError("target model must contain more transformer layers")
    if target.vocab_size < source.vocab_size:
        raise ValueError("target vocabulary cannot be smaller than source vocabulary")
    if embedding_init not in {"mean", "normal", "zero"}:
        raise ValueError("embedding_init must be mean, normal, or zero")

    source_state = source.state_dict()
    target_state = target.state_dict()
    copied = 0
    for name, destination in target_state.items():
        source_value = source_state.get(name)
        if source_value is None:
            if name.startswith("blocks."):
                continue
            raise ValueError(f"target has an unsupported new parameter: {name}")
        if name in _VOCABULARY_PARAMETERS and source_value.shape != destination.shape:
            if (
                source_value.ndim != destination.ndim
                or source_value.shape[1:] != destination.shape[1:]
                or source_value.shape[0] > destination.shape[0]
            ):
                raise ValueError(
                    f"architecture mismatch: incompatible vocabulary parameter shape for {name}"
                )
            replacement = destination.detach().clone()
            rows = source_value.shape[0]
            replacement[:rows].copy_(source_value)
            if rows < replacement.shape[0]:
                if embedding_init == "mean":
                    replacement[rows:].copy_(source_value.mean(dim=0, keepdim=True))
                elif embedding_init == "zero":
                    replacement[rows:].zero_()
                # normal keeps the target model's normal initialization.
            target_state[name] = replacement
            copied += source_value.numel()
            continue
        if source_value.shape != destination.shape:
            raise ValueError(
                f"architecture mismatch for {name}: source={tuple(source_value.shape)} "
                f"target={tuple(destination.shape)}"
            )
        target_state[name] = source_value.detach().clone()
        copied += source_value.numel()

    target.load_state_dict(target_state, strict=True)
    with torch.no_grad():
        for block in target.blocks[source_layers:]:
            block.attn.out_proj.weight.zero_()
            if block.attn.out_proj.bias is not None:
                block.attn.out_proj.bias.zero_()
            block.ffn.out_proj.weight.zero_()
            if block.ffn.out_proj.bias is not None:
                block.ffn.out_proj.bias.zero_()
    if target.tie_word_embeddings:
        target.tie_weights()

    total = target.num_parameters()
    return GrowthReport(
        source_layers=source_layers,
        target_layers=target_layers,
        source_vocab_size=source.vocab_size,
        target_vocab_size=target.vocab_size,
        copied_parameters=copied,
        new_parameters=total - source.num_parameters(),
    )


__all__ = ["GrowthReport", "grow_model"]
