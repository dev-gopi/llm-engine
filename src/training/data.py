"""Configuration-driven training DataLoader construction."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from torch.utils.data import ConcatDataset, DataLoader

from datasets.collator import Collator
from datasets.loader import build_text_dataset
from datasets.token_shards import TokenShardDataset
from datasets.sampler import Sampler
from tokenizer.encoder import Tokenizer


def _mixture_name(path: str | Path, configured: Mapping[str, Any]) -> str:
    """Resolve a weight by filename first, then by dataset directory."""
    source = Path(path)
    filename_name = source.stem.replace("-", "_")
    if filename_name != "train" and filename_name in configured:
        return filename_name
    return source.parent.name.replace("-", "_")


def _mixture_groups(paths: list[str | Path], dataset_sizes: list[int], config: Mapping[str, Any]) -> list[tuple[int, int, float]] | None:
    configured = config.get("dataset_weights")
    if not configured:
        return None
    if not isinstance(configured, Mapping):
        raise ValueError("dataset_weights must be a mapping")
    result: list[tuple[int, int, float]] = []
    start = 0
    for path, size in zip(paths, dataset_sizes, strict=True):
        name = _mixture_name(path, configured)
        weight = float(configured.get(name, 1.0))
        if weight < 0:
            raise ValueError(f"dataset weight must be non-negative: {name}")
        if size:
            result.append((start, start + size, weight))
        start += size
    if not result or not any(weight for _, _, weight in result):
        raise ValueError("dataset_weights must enable at least one non-empty dataset")
    return result


def build_loader(
    paths: Iterable[str | Path],
    tokenizer: Tokenizer,
    config: Mapping[str, Any],
    *,
    shuffle: bool,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    paths = list(paths)
    if paths and all(Path(path).name == "manifest.json" for path in paths):
        shard_datasets = [TokenShardDataset(path) for path in paths]
        expected = int(config.get("max_sequence_length", shard_datasets[0].sequence_length))
        for dataset in shard_datasets:
            if dataset.tokenizer_vocab_size and dataset.tokenizer_vocab_size != tokenizer.vocab_size:
                raise ValueError(
                    f"token shard vocabulary ({dataset.tokenizer_vocab_size}) does not match "
                    f"tokenizer vocabulary ({tokenizer.vocab_size})"
                )
            if (
                dataset.tokenizer_fingerprint is not None
                and dataset.tokenizer_fingerprint != tokenizer.fingerprint
            ):
                raise ValueError(
                    "token shard tokenizer fingerprint does not match the selected tokenizer"
                )
            if dataset.sequence_length != expected:
                raise ValueError(
                    f"token shard sequence length ({dataset.sequence_length}) does not match "
                    f"max_sequence_length ({expected})"
                )
        dataset = shard_datasets[0] if len(shard_datasets) == 1 else ConcatDataset(shard_datasets)
        dataset.lengths = [length for item in shard_datasets for length in item.lengths]
        dataset.dataset_sizes = [len(item) for item in shard_datasets]
    else:
        dataset = build_text_dataset(
            paths, tokenizer, max_length=int(config.get("max_sequence_length", 2048)),
            lazy=bool(config.get("lazy_dataset", True)),
        )
    if not dataset:
        raise ValueError("configured dataset contains no usable examples")
    sampling_groups = _mixture_groups(paths, dataset.dataset_sizes, config) if shuffle and hasattr(dataset, "dataset_sizes") else None
    sampler = Sampler(
        dataset.lengths,
        int(config.get("batch_size", 32)), shuffle=shuffle,
        seed=int(config.get("seed", 42)),
        rank=rank, world_size=world_size,
        sampling_groups=sampling_groups,
        num_samples=int(config.get("samples_per_epoch", len(dataset))) if sampling_groups else None,
    )
    pad_id = tokenizer.token_to_id("<|pad|>")
    if pad_id is None:
        raise ValueError("tokenizer must define <|pad|>")
    import torch
    num_workers = int(config.get("num_workers", 0))
    loader_options: dict[str, Any] = {}
    if num_workers > 0:
        loader_options["persistent_workers"] = bool(config.get("persistent_workers", True))
        loader_options["prefetch_factor"] = int(config.get("prefetch_factor", 2))
        if loader_options["prefetch_factor"] < 1:
            raise ValueError("prefetch_factor must be positive")
    pad_to_multiple_of = config.get("pad_to_multiple_of")
    if pad_to_multiple_of is not None:
        pad_to_multiple_of = int(pad_to_multiple_of)
        if pad_to_multiple_of < 1:
            raise ValueError("pad_to_multiple_of must be positive")
        max_sequence_length = int(config.get("max_sequence_length", 2048))
        if max_sequence_length % pad_to_multiple_of:
            raise ValueError(
                "max_sequence_length must be divisible by pad_to_multiple_of "
                "so padding cannot exceed the model context"
            )
    return DataLoader(
        dataset, batch_sampler=sampler,
        collate_fn=Collator(
            pad_id,
            ignore_index=int(config.get("ignore_index", -100)),
            pad_to_multiple_of=pad_to_multiple_of,
        ),
        num_workers=num_workers,
        pin_memory=bool(config.get("pin_memory", False)) and torch.cuda.is_available(),
        **loader_options,
    )
