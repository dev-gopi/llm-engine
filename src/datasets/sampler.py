"""Deterministic length-aware batch sampling."""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler as TorchSampler


class Sampler(TorchSampler[list[int]]):
    """Shuffle buckets while grouping similar sequence lengths."""

    def __init__(self, lengths: Sequence[int], batch_size: int, *, shuffle: bool = True, drop_last: bool = False, seed: int = 0, bucket_size_multiplier: int = 50, rank: int = 0, world_size: int = 1, sampling_weights: Sequence[float] | None = None, sampling_groups: Sequence[tuple[int, int, float]] | None = None, num_samples: int | None = None) -> None:
        if batch_size < 1 or bucket_size_multiplier < 1:
            raise ValueError("batch and bucket sizes must be positive")
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.bucket_size = batch_size * bucket_size_multiplier
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("rank must be inside a positive world_size")
        self.rank = rank
        self.world_size = world_size
        self.start_batch = 0
        if sampling_weights is not None and sampling_groups is not None:
            raise ValueError("provide sampling_weights or sampling_groups, not both")
        if sampling_weights is not None:
            if len(sampling_weights) != len(self.lengths):
                raise ValueError("sampling_weights must match lengths")
            if not sampling_weights or any(weight < 0 for weight in sampling_weights) or not any(sampling_weights):
                raise ValueError("sampling_weights must contain a positive weight")
            self.sampling_weights = list(sampling_weights)
        else:
            self.sampling_weights = None
        self.sampling_groups = list(sampling_groups or [])
        if self.sampling_groups and (
            any(start < 0 or end <= start or end > len(self.lengths) or weight < 0 for start, end, weight in self.sampling_groups)
            or not any(weight for _, _, weight in self.sampling_groups)
        ):
            raise ValueError("sampling_groups must contain valid ranges and a positive weight")
        self.num_samples = len(self.lengths) if num_samples is None else int(num_samples)
        if self.num_samples < 1:
            raise ValueError("num_samples must be positive")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_start_batch(self, start_batch: int) -> None:
        if start_batch < 0:
            raise ValueError("start_batch must be non-negative")
        self.start_batch = start_batch

    def __iter__(self) -> Iterator[list[int]]:
        randomizer = random.Random(self.seed + self.epoch)
        if self.shuffle and self.sampling_groups:
            target = math.ceil(self.num_samples / self.world_size) * self.world_size
            groups = randomizer.choices(self.sampling_groups, weights=[group[2] for group in self.sampling_groups], k=target)
            indices = [randomizer.randrange(start, end) for start, end, _ in groups]
        elif self.shuffle and self.sampling_weights is not None:
            target = math.ceil(self.num_samples / self.world_size) * self.world_size
            indices = randomizer.choices(range(len(self.lengths)), weights=self.sampling_weights, k=target)
        else:
            indices = list(range(len(self.lengths)))
        if self.shuffle and self.sampling_weights is None:
            randomizer.shuffle(indices)
        if self.world_size > 1:
            if not self.drop_last:
                target = math.ceil(len(indices) / self.world_size) * self.world_size
                indices.extend(indices[: target - len(indices)])
            else:
                indices = indices[: len(indices) - len(indices) % self.world_size]
            indices = indices[self.rank :: self.world_size]
        batches: list[list[int]] = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = sorted(indices[start : start + self.bucket_size], key=self.lengths.__getitem__)
            batches.extend(bucket[index : index + self.batch_size] for index in range(0, len(bucket), self.batch_size))
        if self.drop_last:
            batches = [batch for batch in batches if len(batch) == self.batch_size]
        if self.shuffle:
            randomizer.shuffle(batches)
        yield from batches[self.start_batch :]

    def __len__(self) -> int:
        return max(0, self.total_batches - self.start_batch)

    @property
    def total_batches(self) -> int:
        """Return the full epoch length, independent of a resume offset."""
        total = self.num_samples if self.shuffle and (self.sampling_weights is not None or self.sampling_groups) else len(self.lengths)
        examples = total // self.world_size if self.drop_last else math.ceil(total / self.world_size)
        batches = examples // self.batch_size if self.drop_last else math.ceil(examples / self.batch_size)
        return batches

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "start_batch": self.start_batch}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.epoch = int(state.get("epoch", 0))
        self.set_start_batch(int(state.get("start_batch", 0)))
