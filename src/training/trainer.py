"""Core single-step model trainer."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from model.loss import CausalLanguageModelLoss


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer,
        loss_fn: CausalLanguageModelLoss | None = None,
    ) -> None:
        self.model = model
        self.opt = optimizer
        # Existing trainer callers provide explicit next-token targets, so this
        # integration does not shift them a second time.
        self.loss_fn = loss_fn or CausalLanguageModelLoss(shift_labels=False)

    def train_step(self, inputs: Tensor, targets: Tensor) -> float:
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        logits = self.model(inputs)
        loss = self.loss_fn(logits, targets)
        loss.backward()
        self.opt.step()
        return float(loss.detach().item())
