"""Deterministic length-aware batch sampling."""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler as TorchSampler


class Sampler(TorchSampler[list[int]]):
    """Shuffle buckets while grouping similar sequence lengths."""

    def __init__(self, lengths: Sequence[int], batch_size: int, *, shuffle: bool = True, drop_last: bool = False, seed: int = 0, bucket_size_multiplier: int = 50) -> None:
        if batch_size < 1 or bucket_size_multiplier < 1:
            raise ValueError("batch and bucket sizes must be positive")
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.bucket_size = batch_size * bucket_size_multiplier

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        randomizer = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            randomizer.shuffle(indices)
        batches: list[list[int]] = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = sorted(indices[start : start + self.bucket_size], key=self.lengths.__getitem__)
            batches.extend(bucket[index : index + self.batch_size] for index in range(0, len(bucket), self.batch_size))
        if self.drop_last:
            batches = [batch for batch in batches if len(batch) == self.batch_size]
        if self.shuffle:
            randomizer.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return len(self.lengths) // self.batch_size if self.drop_last else math.ceil(len(self.lengths) / self.batch_size)
