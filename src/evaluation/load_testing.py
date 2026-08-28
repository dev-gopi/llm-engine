"""Deterministic load-test statistics and release-gate evaluation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LoadSample:
    latency_seconds: float
    status_code: int | None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cancelled: bool = False
    error: str | None = None


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between zero and one")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(samples: list[LoadSample], elapsed_seconds: float) -> dict:
    if elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be positive")
    latencies = [sample.latency_seconds for sample in samples if not sample.cancelled]
    successful = [sample for sample in samples if sample.status_code == 200]
    failures = [sample for sample in samples if sample.error or (sample.status_code or 0) >= 500]
    overloads = [sample for sample in samples if sample.status_code in {429, 503}]
    report = {
        "requests": len(samples),
        "successful": len(successful),
        "failed": len(failures),
        "overloaded": len(overloads),
        "cancelled": sum(sample.cancelled for sample in samples),
        "requests_per_second": len(samples) / elapsed_seconds,
        "tokens_per_second": sum(sample.completion_tokens for sample in successful) / elapsed_seconds,
        "latency_seconds": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": max(latencies, default=0.0),
        },
        "samples": [asdict(sample) for sample in samples],
    }
    return report


def release_gate(report: dict, *, max_p95_seconds: float, max_failure_rate: float) -> bool:
    requests = int(report.get("requests", 0))
    if requests < 1:
        return False
    failure_rate = int(report.get("failed", 0)) / requests
    return failure_rate <= max_failure_rate and float(report["latency_seconds"]["p95"]) <= max_p95_seconds
