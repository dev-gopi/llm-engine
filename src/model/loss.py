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
    ) -> None:
        super().__init__()
        self._validate_configuration(
            ignore_index, label_smoothing, z_loss_coefficient, reduction
        )
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
        flat_logits = logits.reshape(-1, vocabulary_size)
        flat_labels = labels.reshape(-1).to(device=logits.device, dtype=torch.long)
        valid_positions = flat_labels.ne(self.ignore_index)
        token_count = int(valid_positions.sum().item())

        if token_count == 0:
            zero = flat_logits.float().sum() * 0.0
            details = LanguageModelLossOutput(zero, zero, zero, 0)
            return details if return_details else zero

        valid_labels = flat_labels[valid_positions]
        if valid_labels.min().item() < 0 or valid_labels.max().item() >= vocabulary_size:
            raise ValueError(f"labels must be in [0, {vocabulary_size}) or equal ignore_index")
        valid_logits = flat_logits[valid_positions].float()

        cross_entropy_sum = F.cross_entropy(
            valid_logits,
            valid_labels,
            reduction="sum",
            label_smoothing=self.label_smoothing,
        )
        if self.z_loss_coefficient:
            log_normalizer = torch.logsumexp(valid_logits, dim=-1)
            z_loss_sum = log_normalizer.square().sum()
        else:
            z_loss_sum = cross_entropy_sum.new_zeros(())

        if self.reduction == "mean":
            cross_entropy = cross_entropy_sum / token_count
            z_loss = z_loss_sum / token_count
        else:
            cross_entropy = cross_entropy_sum
            z_loss = z_loss_sum
        loss = cross_entropy + self.z_loss_coefficient * z_loss
        details = LanguageModelLossOutput(loss, cross_entropy, z_loss, token_count)
        return details if return_details else loss

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "CausalLanguageModelLoss":
        return cls(
            ignore_index=int(config.get("ignore_index", -100)),
            label_smoothing=float(config.get("label_smoothing", 0.0)),
            z_loss_coefficient=float(config.get("z_loss_coefficient", 0.0)),
            shift_labels=bool(config.get("shift_labels", True)),
            reduction=str(config.get("loss_reduction", "mean")),
        )

    def extra_repr(self) -> str:
        return (
            f"ignore_index={self.ignore_index}, label_smoothing={self.label_smoothing}, "
            f"z_loss_coefficient={self.z_loss_coefficient}, "
            f"shift_labels={self.shift_labels}, reduction={self.reduction!r}"
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
