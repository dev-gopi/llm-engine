"""Checkpoint persistence shared by training and inference."""

from __future__ import annotations

import os
import tempfile
import random
from pathlib import Path
from typing import Any

import torch
import numpy as np
from torch import nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    ema: Any = None,
    scaler: Any = None,
    step: int = 0,
    metadata: dict[str, Any] | None = None,
    trainer: dict[str, Any] | None = None,
    sampler: dict[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "ema": ema.state_dict() if ema is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": _numpy_rng_state(),
        "step": int(step),
        "metadata": dict(metadata or {}),
        "trainer": dict(trainer or {}),
        "sampler": dict(sampler or {}),
    }
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return destination


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    ema: Any = None,
    scaler: Any = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    use_ema: bool = False,
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint not found: {source}")
    payload = torch.load(source, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a mapping")
    state = payload.get("model", payload)
    model.load_state_dict(state, strict=strict)
    if use_ema and payload.get("ema") and payload["ema"].get("shadow"):
        parameters = dict(model.named_parameters())
        with torch.no_grad():
            for name, value in payload["ema"]["shadow"].items():
                normalized = name.removeprefix("module.")
                if normalized in parameters:
                    parameters[normalized].copy_(value.to(parameters[normalized].device))
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if ema is not None and payload.get("ema") is not None:
        ema.load_state_dict(payload["ema"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if payload.get("rng_state") is not None:
        torch.set_rng_state(payload["rng_state"].cpu())
    if torch.cuda.is_available() and payload.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    if payload.get("python_rng_state") is not None:
        random.setstate(payload["python_rng_state"])
    if payload.get("numpy_rng_state") is not None:
        _set_numpy_rng_state(payload["numpy_rng_state"])
    return {
        "step": int(payload.get("step", 0)),
        "metadata": payload.get("metadata", {}),
        "trainer": payload.get("trainer", {}),
        "sampler": payload.get("sampler", {}),
        "ema_applied": bool(use_ema and payload.get("ema")),
    }


def _numpy_rng_state() -> dict[str, Any]:
    name, keys, position, gaussian, cached = np.random.get_state()
    return {"name": name, "keys": torch.from_numpy(keys.copy()), "position": position,
            "gaussian": gaussian, "cached": cached}


def _set_numpy_rng_state(state: dict[str, Any]) -> None:
    np.random.set_state((state["name"], state["keys"].cpu().numpy(),
                         int(state["position"]), int(state["gaussian"]), float(state["cached"])))
