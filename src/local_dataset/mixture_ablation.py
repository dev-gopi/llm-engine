"""Deterministic comparison of corpus mixtures against capability metrics."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class MixtureObservation:
    name: str
    mixture: Mapping[str, float]
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.mixture:
            raise ValueError("mixture observations require a name and non-empty mixture")
        if any(not math.isfinite(float(value)) or float(value) < 0 for value in self.mixture.values()):
            raise ValueError("mixture weights must be finite and non-negative")
        if abs(sum(float(value) for value in self.mixture.values()) - 1.0) > 1e-6:
            raise ValueError("mixture weights must sum to 1")
        if any(not name or not math.isfinite(float(value)) for name, value in self.metrics.items()):
            raise ValueError("capability metrics must have non-empty names and finite values")


def compare_mixtures(observations: Sequence[MixtureObservation], baseline: str) -> dict:
    if not observations:
        raise ValueError("at least one mixture observation is required")
    baselines = [item for item in observations if item.name == baseline]
    if len(baselines) != 1:
        raise ValueError("baseline must identify exactly one observation")
    base = baselines[0]
    rows = []
    for item in observations:
        shared = sorted(set(base.metrics) & set(item.metrics))
        rows.append({
            "name": item.name,
            "mixture": dict(item.mixture),
            "metrics": dict(item.metrics),
            "metric_delta_vs_baseline": {name: round(float(item.metrics[name]) - float(base.metrics[name]), 8) for name in shared},
        })
    return {
        "baseline": baseline,
        "observations": len(rows),
        "capability_metrics": sorted(set().union(*(item.metrics.keys() for item in observations))),
        "comparisons": rows,
        "interpretation": "Descriptive metric deltas only; no ranking or causal claim is produced.",
    }
