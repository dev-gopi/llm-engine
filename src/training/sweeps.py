"""Deterministic, config-driven training hyperparameter sweep plans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from typing import Any


_SUPPORTED_AXES = frozenset({
    "learning_rate", "batch_size", "gradient_accumulation_steps", "lr_schedule",
    "optimizer", "weight_decay", "warmup_ratio", "min_lr_ratio",
})


@dataclass(frozen=True)
class SweepTrial:
    """One reproducible training configuration generated from a sweep."""

    name: str
    overrides: dict[str, Any]
    config: dict[str, Any]


def build_sweep(
    base_config: Mapping[str, Any], axes: Mapping[str, Sequence[Any]], *, prefix: str = "trial"
) -> list[SweepTrial]:
    """Expand a Cartesian hyperparameter grid in sorted axis order.

    The result is intentionally a plan: callers must explicitly execute trials.
    This prevents a configuration inspection command from unexpectedly starting
    expensive training work.
    """
    if not prefix or not prefix.replace("_", "").replace("-", "").isalnum():
        raise ValueError("prefix must contain letters, numbers, hyphens, or underscores")
    unknown = set(axes) - _SUPPORTED_AXES
    if unknown:
        raise ValueError(f"unsupported sweep axes: {', '.join(sorted(unknown))}")
    if not axes:
        raise ValueError("at least one sweep axis is required")
    ordered_axes = sorted(axes)
    values = []
    for axis in ordered_axes:
        choices = axes[axis]
        if isinstance(choices, (str, bytes)) or not isinstance(choices, Sequence) or not choices:
            raise ValueError(f"sweep axis {axis!r} must contain at least one value")
        values.append(choices)

    trials: list[SweepTrial] = []
    for index, combination in enumerate(product(*values), start=1):
        overrides = dict(zip(ordered_axes, combination, strict=True))
        if overrides.get("optimizer", "adamw") != "adamw":
            raise ValueError("only the implemented adamw optimizer is supported")
        config = deepcopy(dict(base_config))
        config.update(overrides)
        trials.append(SweepTrial(f"{prefix}-{index:03d}", overrides, config))
    return trials


def sweep_manifest(trials: Sequence[SweepTrial]) -> dict[str, Any]:
    """Return a JSON/YAML-serializable, versioned sweep manifest."""
    return {
        "format_version": 1,
        "trial_count": len(trials),
        "trials": [
            {"name": trial.name, "overrides": trial.overrides, "config": trial.config}
            for trial in trials
        ],
    }
