"""Dynamic padding collator for causal language modeling."""

from __future__ import annotations

import torch
from torch import Tensor


class Collator:
    def __init__(self, pad_token_id: int, *, ignore_index: int = -100, pad_to_multiple_of: int | None = None) -> None:
        if pad_token_id < 0:
            raise ValueError("pad_token_id must be non-negative")
        if pad_to_multiple_of is not None and (
            not isinstance(pad_to_multiple_of, int) or isinstance(pad_to_multiple_of, bool)
            or pad_to_multiple_of < 1
        ):
            raise ValueError("pad_to_multiple_of must be a positive integer")
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, examples: list[dict[str, Tensor] | Tensor]) -> dict[str, Tensor]:
        if not examples:
            raise ValueError("cannot collate an empty batch")
        sequences = [item["input_ids"] if isinstance(item, dict) else item for item in examples]
        if any(not isinstance(sequence, Tensor) or sequence.ndim != 1 or sequence.dtype not in (torch.int32, torch.int64) for sequence in sequences):
            raise ValueError("each input_ids value must be a one-dimensional integer tensor")
        length = max(sequence.numel() for sequence in sequences)
        if self.pad_to_multiple_of:
            length = ((length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        inputs = torch.full((len(sequences), length), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(sequences), length), dtype=torch.bool)
        loss_mask = torch.zeros((len(sequences), length), dtype=torch.bool)
        labels = torch.full_like(inputs, self.ignore_index)
        for row, sequence in enumerate(sequences):
            inputs[row, : sequence.numel()] = sequence.long()
            mask[row, : sequence.numel()] = True
            item = examples[row]
            item = item if isinstance(item, dict) else {}
            for name in ("labels", "attention_mask", "loss_mask"):
                value = item.get(name)
                if value is not None and (not isinstance(value, Tensor) or value.shape != sequence.shape):
                    raise ValueError(f"{name} must have the same shape as input_ids")
                if name in ("attention_mask", "loss_mask") and value is not None:
                    if not torch.all((value == 0) | (value == 1)):
                        raise ValueError(f"{name} must contain binary values")
            example_labels = item.get("labels")
            if example_labels is None:
                example_labels = sequence
            if example_labels.dtype not in (torch.int32, torch.int64):
                raise ValueError("labels must use an integer dtype")
            attention = item.get("attention_mask")
            if attention is not None:
                mask[row, :sequence.numel()] = attention.bool()
            selected = item.get("loss_mask")
            loss_mask[row, :sequence.numel()] = selected.bool() if selected is not None else True
            labels[row, :sequence.numel()] = example_labels.long()
        loss_mask &= mask & labels.ne(self.ignore_index)
        labels.masked_fill_(~loss_mask, self.ignore_index)
        return {"input_ids": inputs, "attention_mask": mask, "labels": labels, "loss_mask": loss_mask}
