"""Preference-pair dataset for DPO-style post-training."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import partial
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from local_dataset.loader import iter_records
from tokenizer.encoder import Tokenizer


class PreferenceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        records: Iterable[Mapping[str, Any]],
        tokenizer: Tokenizer,
        *,
        max_length: int,
    ) -> None:
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        self.examples: list[dict[str, torch.Tensor]] = []
        for record in records:
            if any(not isinstance(record.get(key), str) for key in ("prompt", "chosen", "rejected")):
                continue
            prompt = str(record.get("prompt", "")).strip()
            chosen = str(record.get("chosen", "")).strip()
            rejected = str(record.get("rejected", "")).strip()
            if not prompt or not chosen or not rejected or chosen == rejected:
                continue
            prefix = f"<|user|>\n{prompt}\n<|assistant|>\n"
            prefix_ids = tokenizer.encode(prefix, add_bos=True, allowed_special="all")
            chosen_ids = prefix_ids + tokenizer.encode(chosen, add_eos=True, allowed_special="all")
            rejected_ids = prefix_ids + tokenizer.encode(rejected, add_eos=True, allowed_special="all")
            # Preferences apply to complete answers, not arbitrary truncated prefixes.
            if max(len(chosen_ids), len(rejected_ids)) > max_length:
                continue
            if chosen_ids == rejected_ids:
                continue
            if len(chosen_ids) < 2 or len(rejected_ids) < 2:
                continue
            prompt_tokens = min(len(prefix_ids), max_length)
            self.examples.append({
                "chosen_ids": torch.tensor(chosen_ids),
                "rejected_ids": torch.tensor(rejected_ids),
                "chosen_mask": _response_mask(len(chosen_ids), prompt_tokens),
                "rejected_mask": _response_mask(len(rejected_ids), prompt_tokens),
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


def _response_mask(length: int, prompt_tokens: int) -> torch.Tensor:
    mask = torch.zeros(length - 1, dtype=torch.bool)
    # Logit t predicts token t+1, so response scoring starts one position earlier.
    mask[max(prompt_tokens - 1, 0):] = True
    return mask


def preference_collate(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("preference batch cannot be empty")
    result: dict[str, torch.Tensor] = {}
    for side in ("chosen", "rejected"):
        sequences = [item[f"{side}_ids"] for item in batch]
        masks = [item[f"{side}_mask"] for item in batch]
        width = max(sequence.numel() for sequence in sequences)
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        response_mask = torch.zeros((len(batch), width - 1), dtype=torch.bool)
        attention_mask = torch.zeros((len(batch), width), dtype=torch.bool)
        for row, (sequence, mask) in enumerate(zip(sequences, masks, strict=True)):
            ids[row, :sequence.numel()] = sequence
            response_mask[row, :mask.numel()] = mask
            attention_mask[row, :sequence.numel()] = True
        result[f"{side}_ids"] = ids
        result[f"{side}_mask"] = response_mask
        result[f"{side}_attention_mask"] = attention_mask
    return result


def build_preference_loader(
    paths: Iterable[str], tokenizer: Tokenizer, *, max_length: int,
    batch_size: int, shuffle: bool, seed: int = 42, num_workers: int = 0,
    rank: int = 0, world_size: int = 1,
) -> DataLoader:
    records = (record for path in paths for record in iter_records(path))
    dataset = PreferenceDataset(records, tokenizer, max_length=max_length)
    if not dataset:
        raise ValueError("preference dataset contains no usable chosen/rejected pairs")
    pad_id = tokenizer.token_to_id("<|pad|>")
    if pad_id is None:
        raise ValueError("tokenizer must define <|pad|>")
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError("invalid distributed preference-loader rank/world_size")
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle,
            seed=seed, drop_last=False,
        )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle if sampler is None else False,
        sampler=sampler, generator=generator, num_workers=num_workers,
        collate_fn=partial(preference_collate, pad_id=pad_id),
    )
    loader.gopi_shuffle_seed = seed
    return loader


class KTODataset(Dataset[dict[str, torch.Tensor]]):
    """Binary preference dataset for Kahneman-Tversky Optimization.

    Native records use ``prompt`` + ``completion`` and either a boolean
    ``desirable`` field or ``label`` set to ``desirable``/``undesirable``.
    For migration convenience, classic chosen/rejected pairs are expanded into
    one desirable and one undesirable example without changing their text.
    """

    def __init__(self, records: Iterable[Mapping[str, Any]], tokenizer: Tokenizer, *, max_length: int) -> None:
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        self.examples: list[dict[str, torch.Tensor]] = []
        for record in records:
            expanded: list[tuple[str, str, bool]] = []
            prompt = record.get("prompt")
            if isinstance(prompt, str):
                completion = record.get("completion")
                if isinstance(completion, str):
                    desirable = record.get("desirable")
                    label = record.get("label")
                    if isinstance(desirable, bool):
                        expanded.append((prompt, completion, desirable))
                    elif isinstance(label, str) and label.strip().lower() in {"desirable", "undesirable"}:
                        expanded.append((prompt, completion, label.strip().lower() == "desirable"))
                chosen, rejected = record.get("chosen"), record.get("rejected")
                if isinstance(chosen, str) and isinstance(rejected, str):
                    expanded.extend(((prompt, chosen, True), (prompt, rejected, False)))
            for raw_prompt, raw_completion, desirable in expanded:
                prompt_text, completion_text = raw_prompt.strip(), raw_completion.strip()
                if not prompt_text or not completion_text:
                    continue
                prefix = f"<|user|>\n{prompt_text}\n<|assistant|>\n"
                prefix_ids = tokenizer.encode(prefix, add_bos=True, allowed_special="all")
                ids = prefix_ids + tokenizer.encode(completion_text, add_eos=True, allowed_special="all")
                if len(ids) > max_length or len(ids) < 2:
                    continue
                prompt_tokens = min(len(prefix_ids), max_length)
                self.examples.append({
                    "completion_ids": torch.tensor(ids),
                    "completion_mask": _response_mask(len(ids), prompt_tokens),
                    "desirable": torch.tensor(desirable, dtype=torch.bool),
                })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


def kto_collate(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("KTO batch cannot be empty")
    sequences = [item["completion_ids"] for item in batch]
    masks = [item["completion_mask"] for item in batch]
    width = max(sequence.numel() for sequence in sequences)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    response_mask = torch.zeros((len(batch), width - 1), dtype=torch.bool)
    attention_mask = torch.zeros((len(batch), width), dtype=torch.bool)
    for row, (sequence, mask) in enumerate(zip(sequences, masks, strict=True)):
        ids[row, :sequence.numel()] = sequence
        response_mask[row, :mask.numel()] = mask
        attention_mask[row, :sequence.numel()] = True
    return {
        "completion_ids": ids,
        "completion_mask": response_mask,
        "completion_attention_mask": attention_mask,
        "desirable": torch.stack([item["desirable"] for item in batch]),
    }


def build_kto_loader(
    paths: Iterable[str], tokenizer: Tokenizer, *, max_length: int,
    batch_size: int, shuffle: bool, seed: int = 42, num_workers: int = 0,
    rank: int = 0, world_size: int = 1,
) -> DataLoader:
    records = (record for path in paths for record in iter_records(path))
    dataset = KTODataset(records, tokenizer, max_length=max_length)
    if not dataset:
        raise ValueError("KTO dataset contains no usable desirable/undesirable examples")
    if not any(bool(item["desirable"]) for item in dataset.examples) or not any(
        not bool(item["desirable"]) for item in dataset.examples
    ):
        raise ValueError("KTO dataset must contain both desirable and undesirable examples")
    pad_id = tokenizer.token_to_id("<|pad|>")
    if pad_id is None:
        raise ValueError("tokenizer must define <|pad|>")
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError("invalid distributed KTO-loader rank/world_size")
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=seed, drop_last=False)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle if sampler is None else False,
        sampler=sampler, generator=generator, num_workers=num_workers,
        collate_fn=partial(kto_collate, pad_id=pad_id),
    )
    loader.gopi_shuffle_seed = seed
    return loader


def validate_preference_records(records):
    """Run the alignment provenance/quality gate without changing pair semantics."""
    from alignment.pipeline import audit_preference_set
    result = audit_preference_set(records)
    if result["valid"] != result["records"] or result["duplicates"]:
        raise ValueError(f"preference dataset failed governance: {result}")
    return result
