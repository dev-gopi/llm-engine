"""Offline Group Relative Policy Optimization on pre-scored response groups."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from post_training.dpo import sequence_log_probabilities
from utils.logger import get_logger

logger = get_logger(__name__)


class GRPOLoss(nn.Module):
    def __init__(self, *, clip_epsilon: float = 0.2, beta: float = 0.04, advantage_epsilon: float = 1e-6) -> None:
        super().__init__()
        if not 0 < clip_epsilon < 1:
            raise ValueError("clip_epsilon must be in (0, 1)")
        if beta < 0:
            raise ValueError("beta must be non-negative")
        if advantage_epsilon <= 0:
            raise ValueError("advantage_epsilon must be positive")
        self.clip_epsilon = float(clip_epsilon)
        self.beta = float(beta)
        self.advantage_epsilon = float(advantage_epsilon)

    def forward(
        self,
        policy_logp: Tensor,
        old_logp: Tensor,
        reference_logp: Tensor,
        rewards: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if not (policy_logp.shape == old_logp.shape == reference_logp.shape == rewards.shape):
            raise ValueError("GRPO inputs must share [batch, group] shape")
        if policy_logp.ndim != 2 or policy_logp.shape[1] < 2:
            raise ValueError("GRPO requires at least two completions per prompt group")
        centered = rewards - rewards.mean(dim=1, keepdim=True)
        scale = rewards.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.advantage_epsilon)
        advantages = centered / scale
        log_ratio = policy_logp - old_logp
        ratio = torch.exp(log_ratio.clamp(min=-20.0, max=20.0))
        clipped_ratio = ratio.clamp(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
        surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)
        ref_delta = reference_logp - policy_logp
        kl = torch.exp(ref_delta.clamp(min=-20.0, max=20.0)) - ref_delta - 1.0
        loss = (-surrogate + self.beta * kl).mean()
        clipped = ((ratio < 1.0 - self.clip_epsilon) | (ratio > 1.0 + self.clip_epsilon)).float().mean()
        return loss, {
            "reward_mean": rewards.mean().detach(),
            "reward_std": rewards.std(unbiased=False).detach(),
            "advantage_mean": advantages.mean().detach(),
            "kl": kl.mean().detach(),
            "clip_fraction": clipped.detach(),
        }


class GRPOTrainer:
    def __init__(
        self, policy: nn.Module, old_policy: nn.Module, reference: nn.Module, optimizer, *,
        scheduler=None, clip_epsilon: float = 0.2, beta: float = 0.04,
        gradient_clip_norm: float | None = 1.0, mixed_precision: str = "none",
        distributed_context=None,
    ) -> None:
        self.policy = policy
        try:
            self.device = next(policy.parameters()).device
        except StopIteration as error:
            raise ValueError("policy must contain parameters") from error
        self.old_policy = old_policy.to(self.device).eval()
        self.reference = reference.to(self.device).eval()
        for model in (self.old_policy, self.reference):
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = GRPOLoss(clip_epsilon=clip_epsilon, beta=beta)
        self.gradient_clip_norm = gradient_clip_norm
        if mixed_precision not in {"none", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be none, fp16, or bf16")
        if mixed_precision == "fp16" and self.device.type != "cuda":
            mixed_precision = "none"
        self.mixed_precision = mixed_precision
        self.autocast_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
        self.scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision == "fp16")
        self.distributed_context = distributed_context
        self.global_step = 0
        self.current_epoch = 0
        self.best_validation_loss = float("inf")
        self.epochs_without_improvement = 0
        self.stopped_early = False

    @staticmethod
    def _group_logps(model: nn.Module, values: dict[str, Tensor]) -> Tensor:
        ids = values["completion_ids"]
        batch, group, width = ids.shape
        flat_ids = ids.reshape(batch * group, width)
        flat_attention = values["completion_attention_mask"].reshape(batch * group, width)
        flat_mask = values["completion_mask"].reshape(batch * group, width - 1)
        output = model(flat_ids, attention_mask=flat_attention)
        logits = output[0] if isinstance(output, tuple) else output
        scores = sequence_log_probabilities(logits, flat_ids, flat_mask)
        # Length-normalize to keep response length from dominating group ratios.
        lengths = flat_mask.sum(dim=-1).clamp_min(1).to(scores.dtype)
        return (scores / lengths).reshape(batch, group)

    def _batch_loss(self, values: dict[str, Tensor]):
        policy_logp = self._group_logps(self.policy, values)
        with torch.no_grad():
            old_logp = self._group_logps(self.old_policy, values)
            reference_logp = self._group_logps(self.reference, values)
        return self.loss_fn(policy_logp, old_logp, reference_logp, values["rewards"])

    @staticmethod
    def _batch_size(batch: dict[str, Tensor]) -> int:
        return int(batch["completion_ids"].shape[0])

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
            raise FloatingPointError(f"non-finite GRPO loss at step {self.global_step}")
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        clip_value = self.gradient_clip_norm if self.gradient_clip_norm is not None else float("inf")
        fsdp_clip = getattr(self.policy, "clip_grad_norm_", None)
        gradient_norm = (
            fsdp_clip(clip_value)
            if callable(fsdp_clip)
            else nn.utils.clip_grad_norm_(self.policy.parameters(), clip_value)
        )
        if not bool(torch.isfinite(gradient_norm)):
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.update()
            raise FloatingPointError(f"non-finite GRPO gradients at step {self.global_step}")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()
        self.global_step += 1
        return {"loss": float(loss.detach()), **{key: float(value) for key, value in metrics.items()}}

    @torch.no_grad()
    def evaluate(self, loader) -> dict[str, float]:
        self.policy.eval()
        keys = ("loss", "reward_mean", "reward_std", "advantage_mean", "kl", "clip_fraction")
        totals = {key: 0.0 for key in keys}
        groups = 0
        for batch in loader:
            values = {key: value.to(self.device) for key, value in batch.items()}
            loss, metrics = self._batch_loss(values)
            count = self._batch_size(values)
            totals["loss"] += float(loss) * count
            for key in keys[1:]:
                totals[key] += float(metrics[key]) * count
            groups += count
        if self.distributed_context is not None and self.distributed_context.world_size > 1:
            import torch.distributed as dist
            packed = torch.tensor(
                [*(totals[key] for key in keys), float(groups)],
                device=self.device, dtype=torch.float64,
            )
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            totals = {key: float(packed[index].item()) for index, key in enumerate(keys)}
            groups = int(packed[-1].item())
        if not groups:
            raise ValueError("GRPO validation loader is empty")
        return {key: value / groups for key, value in totals.items()}

    def fit(
        self, train_loader, *, epochs: int, validation_loader=None,
        checkpoint_callback=None, best_checkpoint_callback=None,
        early_stopping_patience: int | None = None, log_every: int = 10,
    ) -> list[dict[str, float | int]]:
        history = []
        keys = ("loss", "reward_mean", "reward_std", "advantage_mean", "kl", "clip_fraction")
        for epoch in range(self.current_epoch, epochs):
            sampler = getattr(train_loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            generator = getattr(train_loader, "generator", None)
            if generator is not None:
                generator.manual_seed(int(getattr(train_loader, "gopi_shuffle_seed", 42)) + epoch)
            totals = {key: 0.0 for key in keys}
            groups = 0
            for batch in train_loader:
                metrics = self.train_step(batch)
                count = self._batch_size(batch)
                groups += count
                for key in keys:
                    totals[key] += metrics[key] * count
                if log_every and self.global_step % log_every == 0:
                    logger.info(
                        "grpo epoch=%d step=%d loss=%.6f reward=%.4f kl=%.5f clip=%.4f",
                        epoch + 1, self.global_step, metrics["loss"], metrics["reward_mean"],
                        metrics["kl"], metrics["clip_fraction"],
                    )
            self.current_epoch = epoch + 1
            if self.distributed_context is not None and self.distributed_context.world_size > 1:
                import torch.distributed as dist
                packed = torch.tensor(
                    [*(totals[key] for key in keys), float(groups)],
                    device=self.device, dtype=torch.float64,
                )
                dist.all_reduce(packed, op=dist.ReduceOp.SUM)
                totals = {key: float(packed[index].item()) for index, key in enumerate(keys)}
                groups = int(packed[-1].item())
            record = {"epoch": epoch + 1, "step": self.global_step, **{
                f"train_{key}": value / max(groups, 1) for key, value in totals.items()
            }}
            if validation_loader is not None:
                validation = self.evaluate(validation_loader)
                record.update({f"validation_{key}": value for key, value in validation.items()})
                if validation["loss"] < self.best_validation_loss:
                    self.best_validation_loss = validation["loss"]
                    self.epochs_without_improvement = 0
                    if best_checkpoint_callback and (
                        self.distributed_context is None or self.distributed_context.is_main_process
                    ):
                        best_checkpoint_callback(self, epoch)
                else:
                    self.epochs_without_improvement += 1
            history.append(record)
            if checkpoint_callback and (
                self.distributed_context is None or self.distributed_context.is_main_process
            ):
                checkpoint_callback(self, epoch)
            if early_stopping_patience is not None and self.epochs_without_improvement >= early_stopping_patience:
                self.stopped_early = True
                break
        return history

    def state_dict(self):
        return {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_validation_loss": self.best_validation_loss,
            "epochs_without_improvement": self.epochs_without_improvement,
            "stopped_early": self.stopped_early,
            "method": "grpo_offline",
        }

    def load_state_dict(self, state) -> None:
        method = state.get("method")
        if method not in {None, "grpo_offline"}:
            raise ValueError(f"checkpoint method {method!r} is not offline GRPO")
        self.global_step = int(state.get("global_step", 0))
        self.current_epoch = int(state.get("current_epoch", 0))
        self.best_validation_loss = float(state.get("best_validation_loss", float("inf")))
        self.epochs_without_improvement = int(state.get("epochs_without_improvement", 0))
        self.stopped_early = bool(state.get("stopped_early", False))
