"""Configuration-driven training DataLoader construction."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
import math
from pathlib import Path
from typing import Any

from torch.utils.data import ConcatDataset, DataLoader

from datasets.collator import Collator
from datasets.loader import build_text_dataset
from datasets.token_shards import TokenShardDataset
from datasets.sampler import Sampler
from tokenizer.encoder import Tokenizer


class InterleavedLoader:
    """Yield batches round-robin so a cap samples every source."""

    def __init__(self, loaders: Iterable[DataLoader]) -> None:
        self.loaders = list(loaders)
        if not self.loaders:
            raise ValueError("interleaved validation requires at least one loader")

    def __len__(self) -> int:
        return sum(len(loader) for loader in self.loaders)

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        active = [iter(loader) for loader in self.loaders]
        while active:
            remaining = []
            for stream in active:
                try:
                    yield next(stream)
                    remaining.append(stream)
                except StopIteration:
                    pass
            active = remaining


def interleave_loaders(loaders: Iterable[DataLoader]):
    loaders = list(loaders)
    return loaders[0] if len(loaders) == 1 else InterleavedLoader(loaders)


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
    names = [_mixture_name(path, configured) for path in paths]
    missing = set(names) - set(configured)
    unused = set(configured) - set(names)
    if missing or unused:
        raise ValueError(
            "dataset_weights must match every training source; "
            f"missing={sorted(missing)}, unused={sorted(unused)}. "
            "Use the dataset directory name or a distinct filename stem."
        )
    result: list[tuple[int, int, float]] = []
    start = 0
    for path, size in zip(paths, dataset_sizes, strict=True):
        name = _mixture_name(path, configured)
        weight = float(configured[name])
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"dataset weight must be finite and non-negative: {name}")
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
    sampler_shuffle: bool | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    paths = list(paths)
    # Check names before indexing potentially multi-gigabyte corpora.
    if shuffle and config.get("dataset_weights"):
        _mixture_groups(paths, [1] * len(paths), config)
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
    resolved_sampler_shuffle = shuffle if sampler_shuffle is None else sampler_shuffle
    sampler = Sampler(
        dataset.lengths,
        int(config.get("batch_size", 32)), shuffle=resolved_sampler_shuffle,
        seed=int(config.get("seed", 42)),
        rank=rank, world_size=world_size,
        sampling_groups=sampling_groups,
        num_samples=int(config.get("samples_per_epoch", len(dataset))) if sampling_groups else None,
    )
    pad_id = tokenizer.token_to_id("<|pad|>")
    if pad_id is None:
        raise ValueError("tokenizer must define <|pad|>")
    import torch
    num_workers = int(config.get("num_workers", 0) if shuffle else
                      config.get("validation_num_workers", config.get("num_workers", 0)))
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    loader_options: dict[str, Any] = {}
    if num_workers > 0:
        loader_options["persistent_workers"] = bool(
            config.get("persistent_workers", True) if shuffle else
            config.get("validation_persistent_workers", config.get("persistent_workers", True))
        )
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
