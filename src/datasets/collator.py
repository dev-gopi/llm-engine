"""Dynamic padding collator for causal language modeling."""

from __future__ import annotations

import torch
from torch import Tensor


class Collator:
    def __init__(self, pad_token_id: int, *, ignore_index: int = -100, pad_to_multiple_of: int | None = None) -> None:
        if pad_token_id < 0:
            raise ValueError("pad_token_id must be non-negative")
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, examples: list[dict[str, Tensor] | Tensor]) -> dict[str, Tensor]:
        if not examples:
            raise ValueError("cannot collate an empty batch")
        sequences = [item["input_ids"] if isinstance(item, dict) else item for item in examples]
        if any(sequence.ndim != 1 or sequence.dtype not in (torch.int32, torch.int64) for sequence in sequences):
            raise ValueError("each input_ids value must be a one-dimensional integer tensor")
        length = max(sequence.numel() for sequence in sequences)
        if self.pad_to_multiple_of:
            length = ((length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        inputs = torch.full((len(sequences), length), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(sequences), length), dtype=torch.bool)
        loss_mask = torch.zeros((len(sequences), length), dtype=torch.bool)
        for row, sequence in enumerate(sequences):
            inputs[row, : sequence.numel()] = sequence.long()
            mask[row, : sequence.numel()] = True
            item = examples[row]
            example_loss_mask = item.get("loss_mask") if isinstance(item, dict) else None
            loss_mask[row, : sequence.numel()] = example_loss_mask.bool() if example_loss_mask is not None else True
        labels = inputs.clone().masked_fill(~loss_mask, self.ignore_index)
        return {"input_ids": inputs, "attention_mask": mask, "labels": labels, "loss_mask": loss_mask}
