"""Numerically stable causal language-model loss calculation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class LanguageModelLossOutput:
    """Detailed loss values for logging and distributed aggregation."""

    loss: Tensor
    cross_entropy: Tensor
    z_loss: Tensor
    token_count: int


class CausalLanguageModelLoss(nn.Module):
    """Cross-entropy objective for next-token prediction.

    Loss arithmetic is performed in float32 even when model logits use FP16 or
    BF16. Labels equal to ``ignore_index`` and positions disabled by
    ``loss_mask`` do not contribute. An optional z-loss penalizes large logit
    normalizers and can improve numerical stability during large-scale training.
    """

    def __init__(
        self,
        *,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        z_loss_coefficient: float = 0.0,
        shift_labels: bool = True,
        reduction: str = "mean",
        chunk_size: int = 0,
    ) -> None:
        super().__init__()
        self._validate_configuration(
            ignore_index, label_smoothing, z_loss_coefficient, reduction
        )
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 0:
            raise ValueError("chunk_size must be a non-negative integer")
        self.chunk_size = chunk_size
        self.ignore_index = ignore_index
        self.label_smoothing = float(label_smoothing)
        self.z_loss_coefficient = float(z_loss_coefficient)
        self.shift_labels = bool(shift_labels)
        self.reduction = reduction

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        *,
        loss_mask: Tensor | None = None,
        return_details: bool = False,
    ) -> Tensor | LanguageModelLossOutput:
        self._validate_inputs(logits, labels, loss_mask)
        if self.shift_labels:
            if logits.size(1) < 2:
                raise ValueError("shifted causal loss requires a sequence length of at least two")
            logits = logits[:, :-1, :]
            labels = labels[:, 1:]
            if loss_mask is not None:
                loss_mask = loss_mask[:, 1:]

        if loss_mask is not None:
            labels = labels.masked_fill(~loss_mask.to(device=labels.device, dtype=torch.bool), self.ignore_index)

        vocabulary_size = logits.size(-1)
        flat_labels = labels.reshape(-1).to(device=logits.device, dtype=torch.long)
        valid_positions = flat_labels.ne(self.ignore_index)
        token_count = int(valid_positions.sum().item())

        if token_count == 0:
            # Avoid reading/casting ignored logits, including NaN/Inf values.
            zero = logits[:, :0, :].float().sum()
            details = LanguageModelLossOutput(zero, zero, zero, 0)
            return details if return_details else zero

        valid_labels = flat_labels[valid_positions]
        if valid_labels.min().item() < 0 or valid_labels.max().item() >= vocabulary_size:
            raise ValueError(f"labels must be in [0, {vocabulary_size}) or equal ignore_index")
        if self.chunk_size and token_count > self.chunk_size:
            # Gather and cast inside the checkpoint so backward retains neither
            # a full selected-logit copy nor vocabulary-sized softmax outputs.
            rows, columns = valid_positions.view(labels.shape).nonzero(as_tuple=True)
            cross_entropy_sum = logits.new_zeros((), dtype=torch.float32)
            z_loss_sum = logits.new_zeros((), dtype=torch.float32)
            for start in range(0, token_count, self.chunk_size):
                end = start + self.chunk_size
                args = (logits, rows[start:end], columns[start:end], valid_labels[start:end])
                if torch.is_grad_enabled() and logits.requires_grad:
                    ce, z = checkpoint(self._selected_loss_sums, *args,
                                       use_reentrant=False, preserve_rng_state=False)
                else:
                    ce, z = self._selected_loss_sums(*args)
                cross_entropy_sum = cross_entropy_sum + ce
                z_loss_sum = z_loss_sum + z
        else:
            if token_count == flat_labels.numel():
                valid_logits = logits.reshape(-1, vocabulary_size)
            else:
                # Select before flattening non-contiguous shifted logits.
                valid_logits = logits[valid_positions.view(labels.shape)]
            cross_entropy_sum, z_loss_sum = self._loss_sums(valid_logits, valid_labels)

        if self.reduction == "mean":
            cross_entropy = cross_entropy_sum / token_count
            z_loss = z_loss_sum / token_count
        else:
            cross_entropy = cross_entropy_sum
            z_loss = z_loss_sum
        loss = cross_entropy + self.z_loss_coefficient * z_loss
        details = LanguageModelLossOutput(loss, cross_entropy, z_loss, token_count)
        return details if return_details else loss

    def _selected_loss_sums(
        self, logits: Tensor, rows: Tensor, columns: Tensor, labels: Tensor
    ) -> tuple[Tensor, Tensor]:
        return self._loss_sums(logits[rows, columns], labels)

    def _loss_sums(self, logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        logits = logits.float()
        ce = F.cross_entropy(logits, labels, reduction="sum",
                             label_smoothing=self.label_smoothing)
        z = (torch.logsumexp(logits, dim=-1).square().sum()
             if self.z_loss_coefficient else ce.new_zeros(()))
        return ce, z

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "CausalLanguageModelLoss":
        return cls(
            ignore_index=int(config.get("ignore_index", -100)),
            label_smoothing=float(config.get("label_smoothing", 0.0)),
            z_loss_coefficient=float(config.get("z_loss_coefficient", 0.0)),
            shift_labels=bool(config.get("shift_labels", True)),
            reduction=str(config.get("loss_reduction", "mean")),
            chunk_size=config.get("loss_chunk_size", 0),
        )

    def extra_repr(self) -> str:
        return (
            f"ignore_index={self.ignore_index}, label_smoothing={self.label_smoothing}, "
            f"z_loss_coefficient={self.z_loss_coefficient}, "
            f"shift_labels={self.shift_labels}, reduction={self.reduction!r}, "
            f"chunk_size={self.chunk_size}"
        )

    @staticmethod
    def _validate_configuration(
        ignore_index: int,
        label_smoothing: float,
        z_loss_coefficient: float,
        reduction: str,
    ) -> None:
        if not isinstance(ignore_index, int) or isinstance(ignore_index, bool):
            raise TypeError("ignore_index must be an integer")
        if not math.isfinite(label_smoothing) or not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must satisfy 0 <= value < 1")
        if not math.isfinite(z_loss_coefficient) or z_loss_coefficient < 0:
            raise ValueError("z_loss_coefficient must be finite and non-negative")
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")

    @staticmethod
    def _validate_inputs(
        logits: Tensor, labels: Tensor, loss_mask: Tensor | None
    ) -> None:
        if not isinstance(logits, Tensor) or not isinstance(labels, Tensor):
            raise TypeError("logits and labels must be torch.Tensor instances")
        if logits.ndim != 3:
            raise ValueError("logits must have shape [batch, sequence, vocabulary]")
        if labels.ndim != 2:
            raise ValueError("labels must have shape [batch, sequence]")
        if logits.shape[:2] != labels.shape:
            raise ValueError("logits and labels batch/sequence dimensions must match")
        if logits.size(-1) < 2:
            raise ValueError("logits vocabulary dimension must contain at least two tokens")
        if not logits.is_floating_point():
            raise TypeError("logits must use a floating-point dtype")
        if labels.dtype not in (torch.int32, torch.int64):
            raise TypeError("labels must use torch.int32 or torch.int64")
        if loss_mask is not None:
            if not isinstance(loss_mask, Tensor):
                raise TypeError("loss_mask must be a torch.Tensor")
            if loss_mask.shape != labels.shape:
                raise ValueError("loss_mask shape must match labels")
