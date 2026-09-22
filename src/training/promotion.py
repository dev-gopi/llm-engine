"""Evidence-based multi-axis checkpoint selection and promotion.

Selection is deterministic and never uses validation loss as the sole criterion.
A checkpoint can only be promoted when required protected capabilities have no
regression beyond configured tolerances and all evidence artifacts are present.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MetricRule:
    name: str
    direction: str = "max"
    weight: float = 1.0
    max_regression: float = 0.0
    protected: bool = False
    def __post_init__(self):
        if self.direction not in {"max", "min"}: raise ValueError("direction must be max or min")
        if self.weight < 0 or self.max_regression < 0: raise ValueError("weight/regression must be non-negative")

@dataclass(frozen=True)
class CheckpointCandidate:
    checkpoint: str
    metrics: Mapping[str, float]
    evidence: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class PromotionDecision:
    selected: str | None
    promoted: bool
    scores: Mapping[str, float]
    failures: tuple[str, ...]

class CheckpointPromoter:
    def __init__(self, rules: list[MetricRule], *, baseline: Mapping[str, float] | None = None):
        if not rules: raise ValueError("at least one metric rule is required")
        if sum(r.weight for r in rules) <= 0: raise ValueError("at least one metric weight must be positive")
        self.rules = tuple(rules); self.baseline = dict(baseline or {})

    def score(self, metrics: Mapping[str, float]) -> float:
        total = sum(r.weight for r in self.rules)
        value = 0.0
        for r in self.rules:
            if r.name not in metrics: raise ValueError(f"missing promotion metric: {r.name}")
            x = float(metrics[r.name])
            # Normalize each axis only relative to baseline when available; otherwise
            # rank candidates by deterministic min/max direction using raw values.
            b = self.baseline.get(r.name)
            if b is not None:
                if b == 0: axis = 1.0 if x == 0 else (0.0 if r.direction == "min" else x)
                elif r.direction == "max": axis = x / b
                else: axis = b / x if x else 0.0
            else:
                axis = x if r.direction == "max" else -x
            value += r.weight * axis
        return value / total

    def _protected_failures(self, candidate: CheckpointCandidate) -> list[str]:
        failures=[]
        for r in self.rules:
            if not r.protected: continue
            if r.name not in candidate.metrics: failures.append(f"missing protected metric:{r.name}"); continue
            if r.name not in self.baseline: continue
            current=float(candidate.metrics[r.name]); base=float(self.baseline[r.name])
            if r.direction == "max": regression = base-current
            else: regression = current-base
            allowed = abs(base) * r.max_regression
            if regression > allowed + 1e-12: failures.append(f"protected regression:{r.name}")
        return failures

    def decide(self, candidates: list[CheckpointCandidate]) -> PromotionDecision:
        if not candidates: return PromotionDecision(None, False, {}, ("no candidates",))
        scores={c.checkpoint:self.score(c.metrics) for c in candidates}
        ordered=sorted(candidates, key=lambda c:(-scores[c.checkpoint], c.checkpoint))
        failures=[]
        for c in ordered:
            missing=[r.name for r in self.rules if r.name not in c.metrics]
            if missing: continue
            pf=self._protected_failures(c)
            if pf: failures.extend([f"{c.checkpoint}:{x}" for x in pf]); continue
            if c.evidence.get("release_blocked", False): failures.append(f"{c.checkpoint}:release_blocked"); continue
            return PromotionDecision(c.checkpoint, True, scores, tuple(failures))
        return PromotionDecision(None, False, scores, tuple(failures or ["no eligible checkpoint"]))
