"""PPO-style RLHF primitives: actor/reference/reward/value models, GAE and clipped PPO."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ValueHead(nn.Module):
    """Scalar value head over the final non-padding hidden state."""
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        self.value_head = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.value_head.weight, std=0.02)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
        if attention_mask is None:
            pooled = hidden_states[:, -1]
        else:
            if attention_mask.shape != hidden_states.shape[:2]:
                raise ValueError("attention_mask must match the first two hidden-state dimensions")
            lengths = attention_mask.to(torch.long).sum(-1)
            if bool((lengths <= 0).any()):
                raise ValueError("value examples must contain a valid token")
            pooled = hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), lengths - 1]
        return self.value_head(pooled).squeeze(-1)


class ValueModel(nn.Module):
    """Decoder backbone plus a trainable scalar critic head."""
    def __init__(self, backbone: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.value_head = ValueHead(hidden_size)

    def forward(self, token_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        output = self.backbone(token_ids, attention_mask=attention_mask, return_hidden_states=True)
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError("value-model backbone must return logits and hidden states")
        return self.value_head(output[1], attention_mask)


def compute_gae(
    rewards: Tensor, values: Tensor, dones: Tensor | None = None, *, gamma: float = 0.99, lam: float = 0.95,
    next_value: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Generalized Advantage Estimation for [batch, time] rollout tensors."""
    if rewards.shape != values.shape:
        raise ValueError("rewards and values must have the same shape")
    if rewards.ndim != 2:
        raise ValueError("rewards and values must have shape [batch, time]")
    if not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError("gamma and lam must be in [0, 1]")
    if dones is None:
        dones = torch.zeros_like(rewards, dtype=torch.bool)
        dones[:, -1] = True
    if dones.shape != rewards.shape:
        raise ValueError("dones must match rewards shape")
    if next_value is None:
        next_value = torch.zeros(rewards.shape[0], device=rewards.device, dtype=rewards.dtype)
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(next_value)
    for t in range(rewards.shape[1] - 1, -1, -1):
        nv = next_value if t == rewards.shape[1] - 1 else values[:, t + 1]
        not_done = (~dones[:, t]).to(rewards.dtype)
        delta = rewards[:, t] + gamma * nv * not_done - values[:, t]
        gae = delta + gamma * lam * not_done * gae
        advantages[:, t] = gae
    returns = advantages + values
    return advantages, returns


def ppo_clipped_loss(
    new_log_probs: Tensor, old_log_probs: Tensor, advantages: Tensor, returns: Tensor,
    values: Tensor, *, old_values: Tensor | None = None, clip_epsilon: float = 0.2, value_clip_epsilon: float | None = 0.2,
    entropy: Tensor | None = None, entropy_coef: float = 0.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute clipped policy and value losses using token-level rollout data."""
    tensors = (new_log_probs, old_log_probs, advantages, returns, values)
    if any(t.shape != new_log_probs.shape for t in tensors[1:]):
        raise ValueError("PPO tensors must share shape")
    if not 0 < clip_epsilon < 1:
        raise ValueError("clip_epsilon must be in (0, 1)")
    if value_clip_epsilon is not None and not 0 < value_clip_epsilon < 1:
        raise ValueError("value_clip_epsilon must be in (0, 1) or None")
    ratio = torch.exp((new_log_probs - old_log_probs).clamp(-20, 20))
    clipped_ratio = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)
    policy_loss = -torch.minimum(ratio * advantages, clipped_ratio * advantages).mean()
    if value_clip_epsilon is None:
        value_loss = 0.5 * F.mse_loss(values, returns)
    else:
        if old_values is None:
            old_values = values.detach()
        if old_values.shape != values.shape:
            raise ValueError("old_values must match values shape")
        clipped_values = old_values + (values - old_values).clamp(-value_clip_epsilon, value_clip_epsilon)
        value_loss = 0.5 * torch.maximum((values - returns).square(), (clipped_values - returns).square()).mean()
    entropy_term = entropy.mean() if entropy is not None else new_log_probs.new_zeros(())
    loss = policy_loss + value_loss - entropy_coef * entropy_term
    return loss, {
        "policy_loss": policy_loss.detach(), "value_loss": value_loss.detach(),
        "clip_fraction": ((ratio < 1 - clip_epsilon) | (ratio > 1 + clip_epsilon)).float().mean().detach(),
        "entropy": entropy_term.detach(),
    }


@dataclass
class AdaptiveKLController:
    """Horizon-based adaptive KL coefficient used by PPO/GRPO."""
    coefficient: float = 0.04
    target: float = 0.1
    horizon: int = 10000
    min_coefficient: float = 1e-6
    max_coefficient: float = 100.0

    def __post_init__(self) -> None:
        if self.coefficient < 0 or self.target <= 0 or self.horizon < 1:
            raise ValueError("invalid adaptive KL controller parameters")
        if not 0 < self.min_coefficient <= self.max_coefficient:
            raise ValueError("invalid KL coefficient bounds")

    def update(self, measured_kl: float, *, steps: int = 1) -> float:
        if measured_kl < 0 or not torch.isfinite(torch.tensor(measured_kl)):
            raise ValueError("measured_kl must be finite and non-negative")
        error = measured_kl / self.target - 1.0
        # PPO's proportional horizon update, bounded to avoid unstable jumps.
        multiplier = 1.0 + max(-0.2, min(0.2, error * steps / self.horizon))
        self.coefficient = min(self.max_coefficient, max(self.min_coefficient, self.coefficient * multiplier))
        return self.coefficient


class PPOTrainer:
    """Minimal but complete actor/reference/reward/value PPO update engine.

    The trainer accepts precomputed rollout tensors, so generation and reward services can
    remain independently scalable. Rollout minibatches can be reused for multiple epochs.
    """
    def __init__(self, actor: nn.Module, value_model: nn.Module, reference: nn.Module,
                 reward_model: nn.Module, optimizer, *, value_optimizer=None, scheduler=None,
                 value_scheduler=None, clip_epsilon: float = 0.2, value_clip_epsilon: float | None = 0.2,
                 gamma: float = 0.99, lam: float = 0.95, kl_controller: AdaptiveKLController | None = None,
                 value_coef: float = 0.5, entropy_coef: float = 0.0, grad_clip: float | None = 1.0) -> None:
        self.actor, self.value_model, self.reference, self.reward_model = actor, value_model, reference, reward_model
        self.optimizer = optimizer
        self.value_optimizer = value_optimizer or optimizer
        self.scheduler, self.value_scheduler = scheduler, value_scheduler
        self.clip_epsilon, self.value_clip_epsilon = float(clip_epsilon), value_clip_epsilon
        self.gamma, self.lam = float(gamma), float(lam)
        self.kl_controller = kl_controller or AdaptiveKLController()
        self.value_coef, self.entropy_coef, self.grad_clip = float(value_coef), float(entropy_coef), grad_clip
        for module in (reference, reward_model):
            module.eval()
            for p in module.parameters(): p.requires_grad_(False)
        self.global_step = 0

    @staticmethod
    def _log_probs(model: nn.Module, token_ids: Tensor, mask: Tensor) -> Tensor:
        from post_training.dpo import sequence_log_probabilities
        out = model(token_ids, attention_mask=(token_ids != 0).to(token_ids.dtype))
        logits = out[0] if isinstance(out, tuple) else out
        return sequence_log_probabilities(logits, token_ids, mask)

    def update(self, rollout: dict[str, Tensor], *, minibatch_size: int | None = None, epochs: int = 1) -> dict[str, float]:
        required = {"token_ids", "old_log_probs", "advantages", "returns"}
        missing = required - set(rollout)
        if missing: raise ValueError(f"PPO rollout missing keys: {sorted(missing)}")
        ids, old_lp = rollout["token_ids"], rollout["old_log_probs"]
        adv, returns = rollout["advantages"], rollout["returns"]
        if not (ids.ndim == 2 and old_lp.ndim == adv.ndim == returns.ndim == 1 and old_lp.shape == adv.shape == returns.shape):
            raise ValueError("token_ids must be [batch,time] and rollout score tensors must share [batch]")
        if minibatch_size is None: minibatch_size = ids.shape[0]
        if minibatch_size < 1 or epochs < 1: raise ValueError("minibatch_size and epochs must be positive")
        n = ids.shape[0]
        totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "kl": 0.0}
        count = 0
        for _ in range(epochs):
            for start in range(0, n, minibatch_size):
                idx = slice(start, min(start + minibatch_size, n))
                batch_ids, batch_old = ids[idx], old_lp[idx]
                mask = rollout.get("mask")
                if mask is None:
                    mask = torch.ones((batch_ids.shape[0], batch_ids.shape[1] - 1), dtype=torch.bool, device=batch_ids.device)
                else:
                    mask = mask[idx]
                new_lp = self._log_probs(self.actor, batch_ids, mask)
                with torch.no_grad():
                    ref_lp = self._log_probs(self.reference, batch_ids, mask)
                # Scalar KL is applied as a reward-side regularizer and adapts online.
                delta = (ref_lp - new_lp).clamp(-20, 20)
                kl = (torch.exp(delta) - delta - 1.0).mean()
                effective_adv = adv[idx] - self.kl_controller.coefficient * (new_lp - ref_lp).detach()
                values = self.value_model(batch_ids, attention_mask=(batch_ids != 0).to(mask.dtype))
                value_tensor = values
                ratio = torch.exp((new_lp - batch_old).clamp(-20, 20))
                clipped = ratio.clamp(1 - self.clip_epsilon, 1 + self.clip_epsilon)
                policy_loss = -torch.minimum(ratio * effective_adv, clipped * effective_adv).mean()
                value_target = returns[idx]
                if value_tensor.shape != value_target.shape:
                    raise ValueError("value model output must match PPO return shape")
                old_values = rollout.get("old_values")
                old_value_tensor = old_values[idx] if old_values is not None else value_tensor.detach()
                if old_value_tensor.shape != value_tensor.shape:
                    raise ValueError("old_values must match value model output")
                clipped_values = old_value_tensor + (value_tensor - old_value_tensor).clamp(-self.value_clip_epsilon, self.value_clip_epsilon) if self.value_clip_epsilon is not None else value_tensor
                value_loss = 0.5 * torch.maximum((value_tensor - value_target).square(), (clipped_values - value_target).square()).mean() if self.value_clip_epsilon is not None else 0.5 * (value_tensor - value_target).square().mean()
                loss = policy_loss + self.value_coef * value_loss
                self.optimizer.zero_grad(set_to_none=True)
                if self.value_optimizer is self.optimizer:
                    loss.backward()
                    if self.grad_clip is not None: nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.value_model.parameters()), self.grad_clip)
                    self.optimizer.step()
                else:
                    policy_loss.backward(retain_graph=True)
                    if self.grad_clip is not None: nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
                    self.optimizer.step()
                    self.value_optimizer.zero_grad(set_to_none=True)
                    (self.value_coef * value_loss).backward()
                    if self.grad_clip is not None: nn.utils.clip_grad_norm_(self.value_model.parameters(), self.grad_clip)
                    self.value_optimizer.step()
                if self.scheduler: self.scheduler.step()
                if self.value_scheduler and self.value_scheduler is not self.scheduler: self.value_scheduler.step()
                self.kl_controller.update(float(kl.detach()), steps=1)
                totals["loss"] += float(loss.detach()); totals["policy_loss"] += float(policy_loss.detach()); totals["value_loss"] += float(value_loss.detach()); totals["kl"] += float(kl.detach()); count += 1
        self.global_step += 1
        return {k: v / max(count, 1) for k, v in totals.items()} | {"kl_coefficient": self.kl_controller.coefficient, "step": float(self.global_step)}


def save_value_checkpoint(path: str | Path, value_model: nn.Module, optimizer=None, *, step: int = 0, metadata: dict | None = None) -> Path:
    dst = Path(path); dst.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": "gopi-value-v1", "step": int(step), "model": value_model.state_dict(), "metadata": metadata or {}}
    if optimizer is not None: payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, dst)
    return dst


def load_value_checkpoint(path: str | Path, value_model: nn.Module, optimizer=None, *, map_location="cpu") -> dict:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("format") != "gopi-value-v1": raise ValueError("unsupported value checkpoint format")
    value_model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload: optimizer.load_state_dict(payload["optimizer"])
    return payload
