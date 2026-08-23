"""Configuration-driven training DataLoader construction."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from datasets.collator import Collator
from datasets.loader import build_text_dataset
from datasets.sampler import Sampler
from tokenizer.encoder import Tokenizer


def build_loader(
    paths: Iterable[str | Path],
    tokenizer: Tokenizer,
    config: Mapping[str, Any],
    *,
    shuffle: bool,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    dataset = build_text_dataset(
        paths, tokenizer, max_length=int(config.get("max_sequence_length", 2048)),
        lazy=bool(config.get("lazy_dataset", True)),
    )
    if not dataset:
        raise ValueError("configured dataset contains no usable examples")
    sampler = Sampler(
        dataset.lengths,
        int(config.get("batch_size", 32)), shuffle=shuffle,
        seed=int(config.get("seed", 42)),
        rank=rank, world_size=world_size,
    )
    pad_id = tokenizer.token_to_id("<|pad|>")
    if pad_id is None:
        raise ValueError("tokenizer must define <|pad|>")
    import torch
    return DataLoader(
        dataset, batch_sampler=sampler,
        collate_fn=Collator(pad_id, ignore_index=int(config.get("ignore_index", -100))),
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=bool(config.get("pin_memory", False)) and torch.cuda.is_available(),
    )
