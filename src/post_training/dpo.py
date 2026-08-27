"""Direct Preference Optimization objective and training step."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from utils.logger import get_logger

logger = get_logger(__name__)


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
    def __init__(
        self, policy: nn.Module, reference: nn.Module, optimizer, *, beta: float = 0.1,
        label_smoothing: float = 0.0, scheduler=None, gradient_clip_norm: float | None = 1.0,
        mixed_precision: str = "none",
    ) -> None:
        self.policy = policy
        try:
            device = next(policy.parameters()).device
        except StopIteration as error:
            raise ValueError("policy must contain parameters") from error
        self.reference = reference.to(device).eval()
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_clip_norm = gradient_clip_norm
        if mixed_precision not in {"none", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be none, fp16, or bf16")
        if mixed_precision == "fp16" and device.type != "cuda":
            mixed_precision = "none"
        self.mixed_precision = mixed_precision
        self.autocast_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
        self.scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision == "fp16")
        self.loss_fn = DPOLoss(beta, label_smoothing)
        self.global_step = 0
        self.current_epoch = 0
        self.best_validation_loss = float("inf")
        self.epochs_without_improvement = 0
        self.stopped_early = False
        for parameter in self.reference.parameters():
            parameter.requires_grad_(False)

    def train_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        values = {key: value.to(self.device) for key, value in batch.items()}
        self.policy.train()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=self.device.type, dtype=self.autocast_dtype,
            enabled=self.mixed_precision != "none",
        ):
            loss, metrics = self._batch_loss(values)
        if not bool(torch.isfinite(loss.detach())):
            raise FloatingPointError(f"non-finite DPO loss at step {self.global_step}")
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        if self.gradient_clip_norm is not None:
            gradient_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.gradient_clip_norm)
            if not bool(torch.isfinite(gradient_norm)):
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
                raise FloatingPointError(f"non-finite DPO gradients at step {self.global_step}")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()
        self.global_step += 1
        return {"loss": float(loss.detach()), **{key: float(value) for key, value in metrics.items()}}

    @torch.no_grad()
    def evaluate(self, loader) -> dict[str, float]:
        self.policy.eval()
        totals = {"loss": 0.0, "reward_accuracy": 0.0, "reward_margin": 0.0}
        batches = 0
        for batch in loader:
            values = {key: value.to(self.device) for key, value in batch.items()}
            loss, metrics = self._batch_loss(values)
            totals["loss"] += float(loss)
            totals["reward_accuracy"] += float(metrics["reward_accuracy"])
            totals["reward_margin"] += float(metrics["reward_margin"])
            batches += 1
        if not batches:
            raise ValueError("DPO validation loader is empty")
        return {key: value / batches for key, value in totals.items()}

    def fit(
        self, train_loader, *, epochs: int, validation_loader=None,
        checkpoint_callback=None, best_checkpoint_callback=None,
        early_stopping_patience: int | None = None, log_every: int = 10,
    ) -> list[dict[str, float | int]]:
        history = []
        for epoch in range(self.current_epoch, epochs):
            generator = getattr(train_loader, "generator", None)
            if generator is not None:
                generator.manual_seed(int(getattr(train_loader, "gopi_shuffle_seed", 42)) + epoch)
            totals = {"loss": 0.0, "reward_accuracy": 0.0, "reward_margin": 0.0}
            batches = 0
            for batch in train_loader:
                metrics = self.train_step(batch)
                batches += 1
                for key in totals:
                    totals[key] += metrics[key]
                if log_every and self.global_step % log_every == 0:
                    logger.info(
                        "dpo epoch=%d step=%d loss=%.6f reward_accuracy=%.4f reward_margin=%.4f",
                        epoch + 1, self.global_step, metrics["loss"],
                        metrics["reward_accuracy"], metrics["reward_margin"],
                    )
            self.current_epoch = epoch + 1
            record = {"epoch": epoch + 1, "step": self.global_step, **{
                f"train_{key}": value / max(batches, 1) for key, value in totals.items()
            }}
            if validation_loader is not None:
                validation = self.evaluate(validation_loader)
                record.update({f"validation_{key}": value for key, value in validation.items()})
                if validation["loss"] < self.best_validation_loss:
                    self.best_validation_loss = validation["loss"]
                    self.epochs_without_improvement = 0
                    if best_checkpoint_callback:
                        best_checkpoint_callback(self, epoch)
                else:
                    self.epochs_without_improvement += 1
            history.append(record)
            if checkpoint_callback:
                checkpoint_callback(self, epoch)
            if early_stopping_patience is not None and self.epochs_without_improvement >= early_stopping_patience:
                self.stopped_early = True
                break
        return history

    def state_dict(self) -> dict[str, int | float | bool]:
        return {
            "global_step": self.global_step, "current_epoch": self.current_epoch,
            "best_validation_loss": self.best_validation_loss,
            "epochs_without_improvement": self.epochs_without_improvement,
            "stopped_early": self.stopped_early,
        }

    def load_state_dict(self, state) -> None:
        self.global_step = int(state.get("global_step", 0))
        self.current_epoch = int(state.get("current_epoch", 0))
        self.best_validation_loss = float(state.get("best_validation_loss", float("inf")))
        self.epochs_without_improvement = int(state.get("epochs_without_improvement", 0))
        self.stopped_early = bool(state.get("stopped_early", False))

    def _batch_loss(self, values: dict[str, Tensor]):
        policy_chosen = self._score(self.policy, values, "chosen")
        policy_rejected = self._score(self.policy, values, "rejected")
        with torch.no_grad():
            reference_chosen = self._score(self.reference, values, "chosen")
            reference_rejected = self._score(self.reference, values, "rejected")
        return self.loss_fn(policy_chosen, policy_rejected, reference_chosen, reference_rejected)

    @staticmethod
    def _score(model: nn.Module, batch: dict[str, Tensor], side: str) -> Tensor:
        ids = batch[f"{side}_ids"]
        output = model(ids, attention_mask=batch[f"{side}_attention_mask"])
        logits = output[0] if isinstance(output, tuple) else output
        return sequence_log_probabilities(logits, ids, batch[f"{side}_mask"])
