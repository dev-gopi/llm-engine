"""Pointwise/regression and listwise reward-model objectives."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def pointwise_regression_loss(
    predictions: Tensor, targets: Tensor, *, loss: str = "huber"
) -> tuple[Tensor, dict[str, Tensor]]:
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have identical shapes")
    if loss == "mse":
        value = F.mse_loss(predictions, targets)
    elif loss == "huber":
        value = F.smooth_l1_loss(predictions, targets)
    else:
        raise ValueError("loss must be mse or huber")
    return value, {"rmse": torch.sqrt(F.mse_loss(predictions, targets).detach())}


def listwise_reward_loss(
    scores: Tensor, target_order: Tensor, *, temperature: float = 1.0
) -> tuple[Tensor, dict[str, Tensor]]:
    if scores.ndim != 2 or target_order.shape != scores.shape:
        raise ValueError("scores and target_order must be [batch,list]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    # Plackett-Luce loss: each position selects the next preferred item.
    loss = scores.new_zeros(())
    for pos in range(scores.shape[1] - 1):
        order = target_order[:, pos:].long()
        row_scores = scores.gather(1, order) / temperature
        loss = loss - (row_scores[:, 0] - torch.logsumexp(row_scores, dim=1)).mean()
    return loss / max(scores.shape[1] - 1, 1), {
        "listwise_loss": loss.detach() / max(scores.shape[1] - 1, 1)
    }
