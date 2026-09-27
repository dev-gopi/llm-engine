"""Pairwise reward-model head and training loop for RLHF/post-training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from utils.logger import get_logger

logger = get_logger(__name__)


class RewardModel(nn.Module):
    """Attach a scalar reward head to a decoder-only language-model backbone.

    The reward is read from the final non-padding token hidden state.  The
    backbone's normal forward API remains unchanged unless
    ``return_hidden_states=True`` is explicitly requested.
    """

    def __init__(self, backbone: nn.Module, hidden_size: int) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.backbone = backbone
        self.reward_head = nn.Linear(hidden_size, 1, bias=True)
        nn.init.normal_(self.reward_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.reward_head.bias)

    def forward(
        self, token_ids: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        output = self.backbone(
            token_ids,
            attention_mask=attention_mask,
            return_hidden_states=True,
        )
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError("reward-model backbone did not return hidden states")
        _, hidden_states = output
        if attention_mask is None:
            indices = torch.full(
                (token_ids.shape[0],),
                token_ids.shape[1] - 1,
                device=token_ids.device,
                dtype=torch.long,
            )
        else:
            if attention_mask.shape != token_ids.shape:
                raise ValueError("attention_mask must match token_ids shape")
            lengths = attention_mask.to(dtype=torch.long).sum(dim=1)
            if bool((lengths <= 0).any()):
                raise ValueError(
                    "reward-model examples must contain at least one valid token"
                )
            indices = lengths - 1
        rows = torch.arange(token_ids.shape[0], device=token_ids.device)
        pooled = hidden_states[rows, indices]
        return self.reward_head(pooled).squeeze(-1)


class PairwiseRewardLoss(nn.Module):
    """Bradley-Terry / logistic pairwise ranking objective."""

    def __init__(self, margin: float = 0.0) -> None:
        super().__init__()
        if margin < 0:
            raise ValueError("margin must be non-negative")
        self.margin = float(margin)

    def forward(
        self, chosen_rewards: Tensor, rejected_rewards: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if chosen_rewards.shape != rejected_rewards.shape:
            raise ValueError(
                "chosen and rejected reward tensors must have identical shapes"
            )
        if chosen_rewards.ndim != 1:
            raise ValueError("reward tensors must be one-dimensional")
        delta = chosen_rewards - rejected_rewards - self.margin
        loss = -F.logsigmoid(delta).mean()
        raw_margin = chosen_rewards - rejected_rewards
        return loss, {
            "reward_accuracy": (raw_margin > 0).float().mean().detach(),
            "reward_margin": raw_margin.mean().detach(),
            "chosen_reward": chosen_rewards.mean().detach(),
            "rejected_reward": rejected_rewards.mean().detach(),
        }


class RewardModelTrainer:
    """Small pairwise reward-model trainer with DDP-compatible reductions."""

    def __init__(
        self,
        model: nn.Module,
        optimizer,
        *,
        scheduler=None,
        margin: float = 0.0,
        gradient_clip_norm: float | None = 1.0,
        mixed_precision: str = "none",
        distributed_context=None,
    ) -> None:
        self.model = model
        try:
            self.device = next(model.parameters()).device
        except StopIteration as error:
            raise ValueError("reward model must contain parameters") from error
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = PairwiseRewardLoss(margin)
        self.gradient_clip_norm = gradient_clip_norm
        if mixed_precision not in {"none", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be none, fp16, or bf16")
        if mixed_precision == "fp16" and self.device.type != "cuda":
            mixed_precision = "none"
        self.mixed_precision = mixed_precision
        self.autocast_dtype = (
            torch.float16 if mixed_precision == "fp16" else torch.bfloat16
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision == "fp16")
        self.distributed_context = distributed_context
        self.global_step = 0
        self.current_epoch = 0
        self.best_validation_loss = float("inf")
        self.epochs_without_improvement = 0
        self.stopped_early = False

    def _batch_loss(self, values: dict[str, Tensor]):
        chosen = self.model(
            values["chosen_ids"], attention_mask=values["chosen_attention_mask"]
        )
        rejected = self.model(
            values["rejected_ids"], attention_mask=values["rejected_attention_mask"]
        )
        return self.loss_fn(chosen, rejected)

    @staticmethod
    def _batch_size(batch: dict[str, Tensor]) -> int:
        return int(batch["chosen_ids"].shape[0])

    def train_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        values = {key: value.to(self.device) for key, value in batch.items()}
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.mixed_precision != "none",
        ):
            loss, metrics = self._batch_loss(values)
        if not bool(torch.isfinite(loss.detach())):
            raise FloatingPointError(
                f"non-finite reward-model loss at step {self.global_step}"
            )
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        gradient_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.gradient_clip_norm
            if self.gradient_clip_norm is not None
            else float("inf"),
        )
        if not bool(torch.isfinite(gradient_norm)):
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.update()
            raise FloatingPointError(
                f"non-finite reward-model gradients at step {self.global_step}"
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()
        self.global_step += 1
        return {
            "loss": float(loss.detach()),
            **{key: float(value) for key, value in metrics.items()},
        }

    @torch.no_grad()
    def evaluate(self, loader) -> dict[str, float]:
        self.model.eval()
        keys = (
            "loss",
            "reward_accuracy",
            "reward_margin",
            "chosen_reward",
            "rejected_reward",
        )
        totals = {key: 0.0 for key in keys}
        pairs = 0
        for batch in loader:
            values = {key: value.to(self.device) for key, value in batch.items()}
            loss, metrics = self._batch_loss(values)
            count = self._batch_size(values)
            totals["loss"] += float(loss) * count
            for key in keys[1:]:
                totals[key] += float(metrics[key]) * count
            pairs += count
        if (
            self.distributed_context is not None
            and self.distributed_context.world_size > 1
        ):
            import torch.distributed as dist

            packed = torch.tensor(
                [*(totals[key] for key in keys), float(pairs)],
                device=self.device,
                dtype=torch.float64,
            )
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            totals = {
                key: float(packed[index].item()) for index, key in enumerate(keys)
            }
            pairs = int(packed[-1].item())
        if not pairs:
            raise ValueError("reward-model validation loader is empty")
        return {key: value / pairs for key, value in totals.items()}

    def fit(
        self,
        train_loader,
        *,
        epochs: int,
        validation_loader=None,
        checkpoint_callback=None,
        best_checkpoint_callback=None,
        early_stopping_patience: int | None = None,
        log_every: int = 10,
    ) -> list[dict[str, float | int]]:
        history: list[dict[str, float | int]] = []
        keys = (
            "loss",
            "reward_accuracy",
            "reward_margin",
            "chosen_reward",
            "rejected_reward",
        )
        for epoch in range(self.current_epoch, epochs):
            sampler = getattr(train_loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            generator = getattr(train_loader, "generator", None)
            if generator is not None:
                generator.manual_seed(
                    int(getattr(train_loader, "gopi_shuffle_seed", 42)) + epoch
                )
            totals = {key: 0.0 for key in keys}
            pairs = 0
            for batch in train_loader:
                metrics = self.train_step(batch)
                count = self._batch_size(batch)
                pairs += count
                for key in keys:
                    totals[key] += metrics[key] * count
                if log_every and self.global_step % log_every == 0:
                    logger.info(
                        "reward_model epoch=%d step=%d loss=%.6f accuracy=%.4f margin=%.4f",
                        epoch + 1,
                        self.global_step,
                        metrics["loss"],
                        metrics["reward_accuracy"],
                        metrics["reward_margin"],
                    )
            self.current_epoch = epoch + 1
            if (
                self.distributed_context is not None
                and self.distributed_context.world_size > 1
            ):
                import torch.distributed as dist

                packed = torch.tensor(
                    [*(totals[key] for key in keys), float(pairs)],
                    device=self.device,
                    dtype=torch.float64,
                )
                dist.all_reduce(packed, op=dist.ReduceOp.SUM)
                totals = {
                    key: float(packed[index].item()) for index, key in enumerate(keys)
                }
                pairs = int(packed[-1].item())
            record: dict[str, float | int] = {
                "epoch": epoch + 1,
                "step": self.global_step,
                **{
                    f"train_{key}": value / max(pairs, 1)
                    for key, value in totals.items()
                },
            }
            if validation_loader is not None:
                validation = self.evaluate(validation_loader)
                record.update(
                    {f"validation_{key}": value for key, value in validation.items()}
                )
                if validation["loss"] < self.best_validation_loss:
                    self.best_validation_loss = validation["loss"]
                    self.epochs_without_improvement = 0
                    if best_checkpoint_callback and (
                        self.distributed_context is None
                        or self.distributed_context.is_main_process
                    ):
                        best_checkpoint_callback(self, epoch)
                else:
                    self.epochs_without_improvement += 1
            history.append(record)
            if checkpoint_callback and (
                self.distributed_context is None
                or self.distributed_context.is_main_process
            ):
                checkpoint_callback(self, epoch)
            if (
                early_stopping_patience is not None
                and self.epochs_without_improvement >= early_stopping_patience
            ):
                self.stopped_early = True
                break
        return history

    def state_dict(self) -> dict[str, int | float | bool]:
        return {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_validation_loss": self.best_validation_loss,
            "epochs_without_improvement": self.epochs_without_improvement,
            "stopped_early": self.stopped_early,
        }

    def load_state_dict(self, state) -> None:
        self.global_step = int(state.get("global_step", 0))
        self.current_epoch = int(state.get("current_epoch", 0))
        self.best_validation_loss = float(
            state.get("best_validation_loss", float("inf"))
        )
        self.epochs_without_improvement = int(
            state.get("epochs_without_improvement", 0)
        )
        self.stopped_early = bool(state.get("stopped_early", False))
