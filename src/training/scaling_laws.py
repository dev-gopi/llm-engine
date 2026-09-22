"""Deterministic summaries for versioned scaling-law experiment observations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ScalingObservation:
    parameters: int
    tokens: int
    validation_loss: float
    capabilities: dict[str, float]

    def __post_init__(self) -> None:
        if self.parameters < 1 or self.tokens < 1 or not math.isfinite(self.validation_loss):
            raise ValueError("parameters, tokens, and validation_loss must be finite positive values")
        if any(not name or not math.isfinite(value) for name, value in self.capabilities.items()):
            raise ValueError("capability metrics must have nonempty names and finite values")

    @property
    def estimated_flops(self) -> float:
        return 6.0 * self.parameters * self.tokens


def summarize_scaling(observations: Sequence[ScalingObservation]) -> dict[str, float | int]:
    """Return deterministic log-compute trend slopes for comparable observations."""
    if len(observations) < 2:
        raise ValueError("at least two observations are required")
    x = [math.log(item.estimated_flops) for item in observations]
    denominator = sum((value - sum(x) / len(x)) ** 2 for value in x)
    if denominator == 0:
        raise ValueError("observations must have distinct compute budgets")

    def slope(values: list[float]) -> float:
        mean_x, mean_y = sum(x) / len(x), sum(values) / len(values)
        return sum((left - mean_x) * (right - mean_y) for left, right in zip(x, values, strict=True)) / denominator

    result: dict[str, float | int] = {"observations": len(observations), "validation_loss_log_compute_slope": slope([item.validation_loss for item in observations])}
    shared = set.intersection(*(set(item.capabilities) for item in observations))
    result.update({f"capability_{name}_log_compute_slope": slope([item.capabilities[name] for item in observations]) for name in sorted(shared)})
    return result
