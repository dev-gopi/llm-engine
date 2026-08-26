"""Direct Preference Optimization objective and training step."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def sequence_log_probabilities(logits: Tensor, token_ids: Tensor, mask: Tensor) -> Tensor:
    if logits.ndim != 3 or token_ids.shape != logits.shape[:2]:
        raise ValueError("logits and token_ids shapes are incompatible")
    if mask.shape != (token_ids.shape[0], token_ids.shape[1] - 1):
        raise ValueError("mask must have shape [batch, sequence - 1]")
    token_logps = F.log_softmax(logits[:, :-1].float(), dim=-1).gather(
        -1, token_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)
    return (token_logps * mask).sum(dim=-1)


class DPOLoss(nn.Module):
    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0) -> None:
        super().__init__()
        if beta <= 0 or not 0 <= label_smoothing < 0.5:
            raise ValueError("beta must be positive and label_smoothing must be in [0, 0.5)")
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(
        self,
        policy_chosen: Tensor,
        policy_rejected: Tensor,
        reference_chosen: Tensor,
        reference_rejected: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        logits = self.beta * (
            (policy_chosen - policy_rejected) - (reference_chosen - reference_rejected)
        )
        losses = (
            -(1 - self.label_smoothing) * F.logsigmoid(logits)
            - self.label_smoothing * F.logsigmoid(-logits)
        )
        rewards_chosen = self.beta * (policy_chosen - reference_chosen).detach()
        rewards_rejected = self.beta * (policy_rejected - reference_rejected).detach()
        return losses.mean(), {
            "reward_accuracy": (rewards_chosen > rewards_rejected).float().mean(),
            "reward_margin": (rewards_chosen - rewards_rejected).mean(),
        }


class DPOTrainer:
    def __init__(self, policy: nn.Module, reference: nn.Module, optimizer, *, beta: float = 0.1) -> None:
        self.policy = policy
        try:
            device = next(policy.parameters()).device
        except StopIteration as error:
            raise ValueError("policy must contain parameters") from error
        self.reference = reference.to(device).eval()
        self.optimizer = optimizer
        self.loss_fn = DPOLoss(beta)
        for parameter in self.reference.parameters():
            parameter.requires_grad_(False)

    def train_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        device = next(self.policy.parameters()).device
        values = {key: value.to(device) for key, value in batch.items()}
        self.policy.train()
        self.optimizer.zero_grad(set_to_none=True)
        policy_chosen = self._score(self.policy, values, "chosen")
        policy_rejected = self._score(self.policy, values, "rejected")
        with torch.no_grad():
            reference_chosen = self._score(self.reference, values, "chosen")
            reference_rejected = self._score(self.reference, values, "rejected")
        loss, metrics = self.loss_fn(
            policy_chosen, policy_rejected, reference_chosen, reference_rejected
        )
        loss.backward()
        self.optimizer.step()
        return {"loss": float(loss.detach()), **{key: float(value) for key, value in metrics.items()}}

    @staticmethod
    def _score(model: nn.Module, batch: dict[str, Tensor], side: str) -> Tensor:
        ids = batch[f"{side}_ids"]
        output = model(ids, attention_mask=batch[f"{side}_attention_mask"])
        logits = output[0] if isinstance(output, tuple) else output
        return sequence_log_probabilities(logits, ids, batch[f"{side}_mask"])
