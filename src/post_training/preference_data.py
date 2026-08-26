"""Preference-pair dataset for DPO-style post-training."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch.utils.data import Dataset

from tokenizer.encoder import Tokenizer


class PreferenceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        records: Iterable[Mapping[str, Any]],
        tokenizer: Tokenizer,
        *,
        max_length: int,
    ) -> None:
        self.examples: list[dict[str, torch.Tensor]] = []
        for record in records:
            prompt = str(record.get("prompt", "")).strip()
            chosen = str(record.get("chosen", "")).strip()
            rejected = str(record.get("rejected", "")).strip()
            if not prompt or not chosen or not rejected or chosen == rejected:
                continue
            prefix = f"<|user|>\n{prompt}\n<|assistant|>\n"
            prefix_ids = tokenizer.encode(prefix, add_bos=True, allowed_special="all")
            chosen_ids = prefix_ids + tokenizer.encode(chosen, add_eos=True, allowed_special="all")
            rejected_ids = prefix_ids + tokenizer.encode(rejected, add_eos=True, allowed_special="all")
            chosen_ids = chosen_ids[:max_length]
            rejected_ids = rejected_ids[:max_length]
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
