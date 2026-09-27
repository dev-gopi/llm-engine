"""Offline group-response dataset for GRPO training."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import partial
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from local_dataset.loader import iter_records
from tokenizer.encoder import Tokenizer

from .preference_data import _response_mask


class GRPODataset(Dataset[dict[str, Any]]):
    """Pre-scored response groups.

    Each record contains ``prompt``, ``completions`` and ``rewards``.  Groups
    remain intact because advantages are normalized within each prompt group.
    """

    def __init__(
        self,
        records: Iterable[Mapping[str, Any]],
        tokenizer: Tokenizer,
        *,
        max_length: int,
    ) -> None:
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        self.examples: list[dict[str, Any]] = []
        for record in records:
            prompt = record.get("prompt")
            completions = record.get("completions")
            rewards = record.get("rewards")
            if (
                not isinstance(prompt, str)
                or not isinstance(completions, list)
                or not isinstance(rewards, list)
            ):
                continue
            prompt = prompt.strip()
            if not prompt or len(completions) < 2 or len(completions) != len(rewards):
                continue
            try:
                numeric_rewards = [float(value) for value in rewards]
            except (TypeError, ValueError):
                continue
            if not all(
                torch.isfinite(torch.tensor(value)) for value in numeric_rewards
            ):
                continue
            prefix = f"<|user|>\n{prompt}\n<|assistant|>\n"
            prefix_ids = tokenizer.encode(prefix, add_bos=True, allowed_special="all")
            group = []
            valid = True
            for completion in completions:
                if not isinstance(completion, str) or not completion.strip():
                    valid = False
                    break
                ids = prefix_ids + tokenizer.encode(
                    completion.strip(), add_eos=True, allowed_special="all"
                )
                if len(ids) < 2 or len(ids) > max_length:
                    valid = False
                    break
                group.append(
                    {
                        "ids": torch.tensor(ids, dtype=torch.long),
                        "mask": _response_mask(
                            len(ids), min(len(prefix_ids), max_length)
                        ),
                    }
                )
            if not valid:
                continue
            # Equal rewards provide no group-relative learning signal.
            if max(numeric_rewards) == min(numeric_rewards):
                continue
            self.examples.append(
                {
                    "group": group,
                    "rewards": torch.tensor(numeric_rewards, dtype=torch.float32),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


def grpo_collate(batch: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("GRPO batch cannot be empty")
    group_sizes = {len(item["group"]) for item in batch}
    if len(group_sizes) != 1:
        raise ValueError(
            "all GRPO groups in a minibatch must have the same number of completions"
        )
    group_size = group_sizes.pop()
    width = max(example["ids"].numel() for item in batch for example in item["group"])
    ids = torch.full((len(batch), group_size, width), pad_id, dtype=torch.long)
    response_mask = torch.zeros((len(batch), group_size, width - 1), dtype=torch.bool)
    attention_mask = torch.zeros((len(batch), group_size, width), dtype=torch.bool)
    rewards = torch.stack([item["rewards"] for item in batch])
    for row, item in enumerate(batch):
        for column, example in enumerate(item["group"]):
            sequence = example["ids"]
            mask = example["mask"]
            ids[row, column, : sequence.numel()] = sequence
            response_mask[row, column, : mask.numel()] = mask
            attention_mask[row, column, : sequence.numel()] = True
    return {
        "completion_ids": ids,
        "completion_mask": response_mask,
        "completion_attention_mask": attention_mask,
        "rewards": rewards,
    }


def build_grpo_loader_from_records(
    records: Iterable[Mapping[str, Any]],
    tokenizer: Tokenizer,
    *,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    seed: int = 42,
    num_workers: int = 0,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """Build a GRPO loader from in-memory groups.

    This is used by online rollout training so newly sampled groups can flow
    directly into the existing, tested GRPO objective without a lossy or
    racy write/read round trip.
    """
    dataset = GRPODataset(records, tokenizer, max_length=max_length)
    if not dataset:
        raise ValueError("GRPO dataset contains no usable pre-scored response groups")
    sizes = {len(item["group"]) for item in dataset.examples}
    if len(sizes) != 1:
        raise ValueError("GRPO dataset must use a consistent group size")
    pad_id = tokenizer.token_to_id("<|pad|>")
    if pad_id is None:
        raise ValueError("tokenizer must define <|pad|>")
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError("invalid distributed GRPO-loader rank/world_size")
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        generator=generator,
        num_workers=num_workers,
        collate_fn=partial(grpo_collate, pad_id=pad_id),
    )
    loader.gopi_shuffle_seed = seed
    return loader


def build_grpo_loader(
    paths: Iterable[str],
    tokenizer: Tokenizer,
    *,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    seed: int = 42,
    num_workers: int = 0,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    records = (record for path in paths for record in iter_records(path))
    return build_grpo_loader_from_records(
        records,
        tokenizer,
        max_length=max_length,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        num_workers=num_workers,
        rank=rank,
        world_size=world_size,
    )
