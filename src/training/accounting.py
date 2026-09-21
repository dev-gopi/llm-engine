"""Reproducible training token, compute, and elapsed-time accounting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingAccounting:
    """Explicit accounting values; FLOPs are an estimate, not a measurement."""

    parameters: int
    supervised_tokens: int
    optimizer_steps: int
    elapsed_seconds: float
    device_count: int = 1
    flop_multiplier: float = 6.0

    def __post_init__(self) -> None:
        if self.parameters < 1 or self.supervised_tokens < 0 or self.optimizer_steps < 0:
            raise ValueError("parameters and counters must be non-negative, with parameters positive")
        if self.elapsed_seconds < 0 or self.device_count < 1 or self.flop_multiplier <= 0:
            raise ValueError("elapsed_seconds, device_count, and flop_multiplier must be positive")

    @property
    def estimated_flops(self) -> float:
        """Use the conventional 6 × parameters × training-tokens estimate."""
        return self.flop_multiplier * self.parameters * self.supervised_tokens

    @property
    def device_hours(self) -> float:
        return self.elapsed_seconds * self.device_count / 3600

    @property
    def tokens_per_second(self) -> float:
        return self.supervised_tokens / self.elapsed_seconds if self.elapsed_seconds else 0.0

    def report(self) -> dict[str, int | float]:
        return {
            "parameters": self.parameters,
            "supervised_tokens": self.supervised_tokens,
            "optimizer_steps": self.optimizer_steps,
            "elapsed_seconds": self.elapsed_seconds,
            "device_count": self.device_count,
            "estimated_flops": self.estimated_flops,
            "device_hours": self.device_hours,
            "tokens_per_second": self.tokens_per_second,
            "flop_estimate_multiplier": self.flop_multiplier,
        }
