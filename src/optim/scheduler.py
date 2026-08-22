"""Warmup with cosine/linear/constant learning-rate decay."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class Scheduler(LambdaLR):
    def __init__(self, optimizer: Optimizer, *, warmup_steps: int, total_steps: int, schedule: str = "cosine", min_lr_ratio: float = 0.1, last_epoch: int = -1) -> None:
        if not 0 <= warmup_steps < total_steps:
            raise ValueError("warmup_steps must satisfy 0 <= warmup_steps < total_steps")
        if schedule not in {"cosine", "linear", "constant"}:
            raise ValueError("schedule must be cosine, linear, or constant")
        if not 0 <= min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be between zero and one")
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.schedule = schedule
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, self._multiplier, last_epoch=last_epoch)

    def _multiplier(self, step: int) -> float:
        if self.warmup_steps and step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        if self.schedule == "constant":
            return 1.0
        progress = min(1.0, max(0.0, (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)))
        decay = 1.0 - progress if self.schedule == "linear" else 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * decay

    @classmethod
    def from_config(cls, optimizer: Optimizer, config: Mapping[str, Any], *, total_steps: int | None = None) -> "Scheduler":
        resolved_total = total_steps or int(config["total_steps"])
        warmup = config.get("warmup_steps")
        if warmup is None:
            warmup = round(resolved_total * float(config.get("warmup_ratio", 0.03)))
        return cls(
            optimizer, warmup_steps=int(warmup), total_steps=resolved_total,
            schedule=str(config.get("lr_schedule", "cosine")),
            min_lr_ratio=float(config.get("min_lr_ratio", 0.1)),
        )
