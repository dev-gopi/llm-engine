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




def sequence_average_log_probabilities(logits: Tensor, token_ids: Tensor, mask: Tensor) -> Tensor:
    summed = sequence_log_probabilities(logits, token_ids, mask)
    counts = mask.sum(dim=-1).clamp_min(1).to(summed.dtype)
    return summed / counts


def _log1mexp(log_p: Tensor) -> Tensor:
    """Stable log(1-exp(log_p)) for log probabilities <= 0."""
    log_p = torch.clamp(log_p, max=-1e-7)
    cutoff = -0.6931471805599453
    return torch.where(
        log_p < cutoff,
        torch.log1p(-torch.exp(log_p)),
        torch.log(-torch.expm1(log_p)),
    )


class ORPOLoss(nn.Module):
    """Odds Ratio Preference Optimization on paired chosen/rejected responses.

    The objective combines chosen-response negative log likelihood with an
    odds-ratio preference term. Inputs are average response log probabilities.
    """

    def __init__(self, preference_weight: float = 0.1) -> None:
        super().__init__()
        if preference_weight <= 0:
            raise ValueError("preference_weight must be positive")
        self.preference_weight = float(preference_weight)

    def forward(self, chosen_logp: Tensor, rejected_logp: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        chosen_log_odds = chosen_logp - _log1mexp(chosen_logp)
        rejected_log_odds = rejected_logp - _log1mexp(rejected_logp)
        log_odds_ratio = chosen_log_odds - rejected_log_odds
        chosen_nll = -chosen_logp.mean()
        preference_loss = -F.logsigmoid(log_odds_ratio).mean()
        loss = chosen_nll + self.preference_weight * preference_loss
        return loss, {
            "reward_accuracy": (chosen_logp > rejected_logp).float().mean(),
            "reward_margin": (chosen_logp - rejected_logp).mean().detach(),
        }


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


class IPOLoss(nn.Module):
    """Identity Preference Optimization loss on paired preferences.

    IPO fits the policy/reference log-ratio gap to the closed-form target
    ``1 / (2 * beta)`` from the IPO objective. The metric definitions mirror
    DPO so existing reports and release gates remain comparable.
    """

    def __init__(self, beta: float = 0.1) -> None:
        super().__init__()
        if beta <= 0:
            raise ValueError("beta must be positive")
        self.beta = beta

    def forward(
        self,
        policy_chosen: Tensor,
        policy_rejected: Tensor,
        reference_chosen: Tensor,
        reference_rejected: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        log_ratio = (policy_chosen - policy_rejected) - (
            reference_chosen - reference_rejected
        )
        target = 1.0 / (2.0 * self.beta)
        losses = (log_ratio - target).square()
        rewards_chosen = self.beta * (policy_chosen - reference_chosen).detach()
        rewards_rejected = self.beta * (policy_rejected - reference_rejected).detach()
        return losses.mean(), {
            "reward_accuracy": (rewards_chosen > rewards_rejected).float().mean(),
            "reward_margin": (rewards_chosen - rewards_rejected).mean(),
        }


class KTOLoss(nn.Module):
    """Kahneman-Tversky Optimization for binary desirable/undesirable feedback.

    Rewards are policy/reference log-ratios scaled by ``beta``. The batch KL
    baseline is detached and clamped non-negative, matching the KTO utility
    construction while keeping the implementation stable for small batches.
    """

    def __init__(self, beta: float = 0.1, desirable_weight: float = 1.0, undesirable_weight: float = 1.0) -> None:
        super().__init__()
        if beta <= 0:
            raise ValueError("beta must be positive")
        if desirable_weight <= 0 or undesirable_weight <= 0:
            raise ValueError("KTO class weights must be positive")
        self.beta = float(beta)
        self.desirable_weight = float(desirable_weight)
        self.undesirable_weight = float(undesirable_weight)

    def forward(
        self, policy_logp: Tensor, reference_logp: Tensor, desirable: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if policy_logp.shape != reference_logp.shape or desirable.shape != policy_logp.shape:
            raise ValueError("KTO inputs must have matching batch shapes")
        desirable = desirable.bool()
        if desirable.numel() == 0:
            raise ValueError("KTO batch cannot be empty")
        log_ratio = policy_logp - reference_logp
        kl = log_ratio.detach().mean().clamp_min(0.0)
        reward = self.beta * log_ratio
        baseline = self.beta * kl
        pieces = []
        if bool(desirable.any()):
            pieces.append(self.desirable_weight * (1.0 - torch.sigmoid(reward[desirable] - baseline)))
        if bool((~desirable).any()):
            pieces.append(self.undesirable_weight * (1.0 - torch.sigmoid(baseline - reward[~desirable])))
        weighted = torch.cat(pieces)
        signed_margin = torch.where(desirable, reward - baseline, baseline - reward)
        accuracy = torch.where(desirable, reward > baseline, reward < baseline).float().mean()
        return weighted.mean(), {
            "reward_accuracy": accuracy.detach(),
            "reward_margin": signed_margin.mean().detach(),
        }


class DPOTrainer:
    def __init__(
        self, policy: nn.Module, reference: nn.Module | None, optimizer, *, beta: float = 0.1,
        label_smoothing: float = 0.0, orpo_lambda: float = 0.1, kto_desirable_weight: float = 1.0, kto_undesirable_weight: float = 1.0, scheduler=None,
        gradient_clip_norm: float | None = 1.0, mixed_precision: str = "none",
        method: str = "dpo", distributed_context=None,
    ) -> None:
        self.policy = policy
        try:
            device = next(policy.parameters()).device
        except StopIteration as error:
            raise ValueError("policy must contain parameters") from error
        self.reference = reference.to(device).eval() if reference is not None else None
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
        method = str(method).lower()
        if method not in {"dpo", "ipo", "orpo", "kto"}:
            raise ValueError("preference method must be dpo, ipo, orpo, or kto")
        if method != "dpo" and label_smoothing:
            raise ValueError("label_smoothing is only supported for DPO")
        if method in {"dpo", "ipo", "kto"} and self.reference is None:
            raise ValueError(f"{method.upper()} requires a reference model")
        self.method = method
        if method == "dpo":
            self.loss_fn = DPOLoss(beta, label_smoothing)
        elif method == "ipo":
            self.loss_fn = IPOLoss(beta)
        elif method == "orpo":
            self.loss_fn = ORPOLoss(orpo_lambda)
        else:
            self.loss_fn = KTOLoss(beta, kto_desirable_weight, kto_undesirable_weight)
        self.distributed_context = distributed_context
        self.global_step = 0
        self.current_epoch = 0
        self.best_validation_loss = float("inf")
        self.epochs_without_improvement = 0
        self.stopped_early = False
        if self.reference is not None:
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
        gradient_norm = nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            self.gradient_clip_norm if self.gradient_clip_norm is not None else float("inf"),
        )
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
        pairs = 0
        for batch in loader:
            values = {key: value.to(self.device) for key, value in batch.items()}
            loss, metrics = self._batch_loss(values)
            count = self._batch_size(values)
            totals["loss"] += float(loss) * count
            totals["reward_accuracy"] += float(metrics["reward_accuracy"]) * count
            totals["reward_margin"] += float(metrics["reward_margin"]) * count
            pairs += count
        if self.distributed_context is not None and self.distributed_context.world_size > 1:
            import torch.distributed as dist
            packed = torch.tensor(
                [totals["loss"], totals["reward_accuracy"], totals["reward_margin"], float(pairs)],
                device=self.device, dtype=torch.float64,
            )
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            totals = {
                "loss": float(packed[0].item()),
                "reward_accuracy": float(packed[1].item()),
                "reward_margin": float(packed[2].item()),
            }
            pairs = int(packed[3].item())
        if not pairs:
            raise ValueError("preference validation loader is empty")
        return {key: value / pairs for key, value in totals.items()}

    def fit(
        self, train_loader, *, epochs: int, validation_loader=None,
        checkpoint_callback=None, best_checkpoint_callback=None,
        early_stopping_patience: int | None = None, log_every: int = 10,
    ) -> list[dict[str, float | int]]:
        history = []
        for epoch in range(self.current_epoch, epochs):
            sampler = getattr(train_loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            generator = getattr(train_loader, "generator", None)
            if generator is not None:
                generator.manual_seed(int(getattr(train_loader, "gopi_shuffle_seed", 42)) + epoch)
            totals = {"loss": 0.0, "reward_accuracy": 0.0, "reward_margin": 0.0}
            pairs = 0
            for batch in train_loader:
                metrics = self.train_step(batch)
                count = self._batch_size(batch)
                pairs += count
                for key in totals:
                    totals[key] += metrics[key] * count
                if log_every and self.global_step % log_every == 0:
                    logger.info(
                        "dpo epoch=%d step=%d loss=%.6f reward_accuracy=%.4f reward_margin=%.4f",
                        epoch + 1, self.global_step, metrics["loss"],
                        metrics["reward_accuracy"], metrics["reward_margin"],
                    )
            self.current_epoch = epoch + 1
            if self.distributed_context is not None and self.distributed_context.world_size > 1:
                import torch.distributed as dist
                packed = torch.tensor(
                    [totals["loss"], totals["reward_accuracy"], totals["reward_margin"], float(pairs)],
                    device=self.device, dtype=torch.float64,
                )
                dist.all_reduce(packed, op=dist.ReduceOp.SUM)
                totals = {
                    "loss": float(packed[0].item()),
                    "reward_accuracy": float(packed[1].item()),
                    "reward_margin": float(packed[2].item()),
                }
                pairs = int(packed[3].item())
            record = {"epoch": epoch + 1, "step": self.global_step, **{
                f"train_{key}": value / max(pairs, 1) for key, value in totals.items()
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

    def state_dict(self) -> dict[str, int | float | bool]:
        return {
            "global_step": self.global_step, "current_epoch": self.current_epoch,
            "best_validation_loss": self.best_validation_loss,
            "epochs_without_improvement": self.epochs_without_improvement,
            "stopped_early": self.stopped_early, "method": self.method,
        }

    def load_state_dict(self, state) -> None:
        saved_method = state.get("method")
        if saved_method is not None and str(saved_method).lower() != self.method:
            raise ValueError(
                f"checkpoint preference method {saved_method!r} does not match active method {self.method!r}"
            )
        self.global_step = int(state.get("global_step", 0))
        self.current_epoch = int(state.get("current_epoch", 0))
        self.best_validation_loss = float(state.get("best_validation_loss", float("inf")))
        self.epochs_without_improvement = int(state.get("epochs_without_improvement", 0))
        self.stopped_early = bool(state.get("stopped_early", False))

    def _batch_loss(self, values: dict[str, Tensor]):
        if self.method == "kto":
            policy_logp = self._score_completion(self.policy, values)
            assert self.reference is not None
            with torch.no_grad():
                reference_logp = self._score_completion(self.reference, values)
            return self.loss_fn(policy_logp, reference_logp, values["desirable"])
        if self.method == "orpo":
            chosen = self._score(self.policy, values, "chosen", average=True)
            rejected = self._score(self.policy, values, "rejected", average=True)
            return self.loss_fn(chosen, rejected)
        policy_chosen = self._score(self.policy, values, "chosen")
        policy_rejected = self._score(self.policy, values, "rejected")
        assert self.reference is not None
        with torch.no_grad():
            reference_chosen = self._score(self.reference, values, "chosen")
            reference_rejected = self._score(self.reference, values, "rejected")
        return self.loss_fn(policy_chosen, policy_rejected, reference_chosen, reference_rejected)

    @staticmethod
    def _batch_size(batch: dict[str, Tensor]) -> int:
        key = "completion_ids" if "completion_ids" in batch else "chosen_ids"
        return int(batch[key].shape[0])

    @staticmethod
    def _score_completion(model: nn.Module, batch: dict[str, Tensor]) -> Tensor:
        ids = batch["completion_ids"]
        output = model(ids, attention_mask=batch["completion_attention_mask"])
        logits = output[0] if isinstance(output, tuple) else output
        return sequence_log_probabilities(logits, ids, batch["completion_mask"])

    @staticmethod
    def _score(model: nn.Module, batch: dict[str, Tensor], side: str, *, average: bool = False) -> Tensor:
        ids = batch[f"{side}_ids"]
        output = model(ids, attention_mask=batch[f"{side}_attention_mask"])
        logits = output[0] if isinstance(output, tuple) else output
        scorer = sequence_average_log_probabilities if average else sequence_log_probabilities
        return scorer(logits, ids, batch[f"{side}_mask"])


def compare_sft_to_dpo(sft_metrics: dict[str,float], dpo_metrics: dict[str,float]) -> dict[str,object]:
    """Return capability/regression deltas; callers decide release policy."""
    keys=sorted(set(sft_metrics).intersection(dpo_metrics))
    return {"sft":dict(sft_metrics),"dpo":dict(dpo_metrics),"delta":{k:float(dpo_metrics[k])-float(sft_metrics[k]) for k in keys}}
