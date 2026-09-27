"""Reward-model calibration drift monitoring and safe version promotion."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import math

@dataclass(frozen=True)
class RewardCalibrationVersion:
    version: str
    threshold: float
    mean: float
    std: float
    created_at: str

@dataclass
class RewardDriftMonitor:
    alert_threshold: float = 0.2
    versions: list[RewardCalibrationVersion] = field(default_factory=list)
    promoted_version: str | None = None
    def observe(self, version: str, scores: list[float], *, threshold: float | None = None) -> dict[str, Any]:
        if not scores: raise ValueError("scores cannot be empty")
        if any(not math.isfinite(float(x)) for x in scores): raise ValueError("scores must be finite")
        mean = sum(scores) / len(scores); variance = sum((x-mean)**2 for x in scores)/len(scores); std = variance**0.5
        ref = self.versions[-1] if self.versions else None
        drift = 0.0 if ref is None else abs(mean-ref.mean) / max(ref.std, 1e-6)
        item = RewardCalibrationVersion(version, float(threshold if threshold is not None else (ref.threshold if ref else 0.0)), mean, std, datetime.now(timezone.utc).isoformat())
        self.versions.append(item)
        return {"version": version, "mean": mean, "std": std, "drift": drift, "alert": drift >= self.alert_threshold}
    def promote(self, version: str) -> RewardCalibrationVersion:
        match = next((x for x in self.versions if x.version == version), None)
        if match is None: raise ValueError(f"unknown reward version {version!r}")
        self.promoted_version = version; return match
    def rollback(self, version: str) -> RewardCalibrationVersion: return self.promote(version)
