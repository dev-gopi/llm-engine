"""Checkpoint persistence shared by training and inference."""

from __future__ import annotations

import os
import tempfile
import random
from collections.abc import Collection
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
    restore_rng: bool = True,
    expected_tokenizer_fingerprint: str | None = None,
    compatible_tokenizer_fingerprints: Collection[str] = (),
    allow_vocab_extension: bool = False,
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint not found: {source}")
    payload = torch.load(source, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a mapping")
    saved_fingerprint = payload.get("metadata", {}).get("tokenizer_fingerprint")
    if (
        expected_tokenizer_fingerprint is not None
        and saved_fingerprint is not None
        and saved_fingerprint != expected_tokenizer_fingerprint
        and saved_fingerprint not in compatible_tokenizer_fingerprints
    ):
        raise ValueError(
            "checkpoint tokenizer fingerprint does not match the selected tokenizer; "
            "use the tokenizer that trained this checkpoint or a verified append-only extension"
        )
    state = payload.get("model", payload)
    if allow_vocab_extension:
        state = _expand_vocabulary_state(state, model.state_dict())
    try:
        model.load_state_dict(state, strict=strict)
    except RuntimeError as err:
        saved_config = payload.get("metadata", {}).get("model_config")
        if saved_config:
            raise RuntimeError(
                f"Failed to load checkpoint '{source}' due to model architecture mismatch. "
                f"Checkpoint model config: {saved_config}. Original error: {err}"
            ) from err
        raise RuntimeError(
            f"Failed to load checkpoint '{source}' due to model architecture mismatch. "
            f"Please verify that --model-config matches the architecture used when creating the checkpoint. "
            f"Original error: {err}"
        ) from err
    if use_ema and payload.get("ema") and payload["ema"].get("shadow"):
        parameters = dict(model.named_parameters())
        with torch.no_grad():
            for name, value in payload["ema"]["shadow"].items():
                normalized = name.removeprefix("module.")
                if normalized in parameters:
                    parameter = parameters[normalized]
                    source_value = value.to(parameter.device)
                    if (
                        allow_vocab_extension
                        and normalized in _VOCABULARY_PARAMETERS
                        and source_value.ndim == parameter.ndim
                        and source_value.shape[1:] == parameter.shape[1:]
                        and source_value.shape[0] < parameter.shape[0]
                    ):
                        parameter[: source_value.shape[0]].copy_(source_value)
                        parameter[source_value.shape[0] :].copy_(
                            _vocabulary_row_mean(source_value)
                        )
                    else:
                        parameter.copy_(source_value)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if ema is not None and payload.get("ema") is not None:
        ema.load_state_dict(payload["ema"])
    # A disabled GradScaler serializes as an empty mapping. This is a valid
    # checkpoint state (for example BF16/CPU -> FP16 GPU), but an enabled
    # GradScaler rejects it. In that case retain the new scaler's fresh state.
    if scaler is not None and payload.get("scaler"):
        scaler.load_state_dict(payload["scaler"])
    if restore_rng:
        if payload.get("rng_state") is not None:
            try:
                torch.set_rng_state(payload["rng_state"].cpu().to(torch.uint8))
            except Exception:
                pass
        if torch.cuda.is_available() and payload.get("cuda_rng_state") is not None:
            try:
                cuda_states = [
                    s.cpu().to(torch.uint8) if isinstance(s, torch.Tensor) else s
                    for s in payload["cuda_rng_state"]
                ]
                if len(cuda_states) == torch.cuda.device_count():
                    torch.cuda.set_rng_state_all(cuda_states)
            except Exception:
                pass
        if payload.get("python_rng_state") is not None:
            try:
                random.setstate(payload["python_rng_state"])
            except Exception:
                pass
        if payload.get("numpy_rng_state") is not None:
            try:
                _set_numpy_rng_state(payload["numpy_rng_state"])
            except Exception:
                pass
    return {
        "step": int(payload.get("step", 0)),
        "metadata": payload.get("metadata", {}),
        "trainer": payload.get("trainer", {}),
        "sampler": payload.get("sampler", {}),
        "ema_applied": bool(use_ema and payload.get("ema")),
    }


_VOCABULARY_PARAMETERS = ("tok.embedding.weight", "head.weight", "head.bias")


def _expand_vocabulary_state(
    saved_state: dict[str, Any], target_state: dict[str, Any]
) -> dict[str, Any]:
    """Pad checkpoint vocabulary tensors with the model's initialized rows."""
    expanded = dict(saved_state)
    for name, saved in saved_state.items():
        normalized = name.removeprefix("module.")
        if normalized not in _VOCABULARY_PARAMETERS or name not in target_state:
            continue
        target = target_state[name]
        if not isinstance(saved, torch.Tensor) or saved.shape == target.shape:
            continue
        if (
            saved.ndim != target.ndim
            or saved.shape[1:] != target.shape[1:]
            or saved.shape[0] >= target.shape[0]
        ):
            continue
        replacement = target.detach().clone()
        replacement[: saved.shape[0]].copy_(saved.to(replacement.device))
        replacement[saved.shape[0] :].copy_(
            _vocabulary_row_mean(saved.to(replacement.device))
        )
        expanded[name] = replacement
    return expanded


def _vocabulary_row_mean(values: torch.Tensor) -> torch.Tensor:
    """Return a broadcastable centroid for stable append-only initialization."""
    return values.float().mean(dim=0, keepdim=True).to(dtype=values.dtype)


def _numpy_rng_state() -> dict[str, Any]:
    name, keys, position, gaussian, cached = np.random.get_state()
    return {"name": name, "keys": torch.from_numpy(keys.copy()), "position": position,
            "gaussian": gaussian, "cached": cached}


def _set_numpy_rng_state(state: dict[str, Any]) -> None:
    np.random.set_state((state["name"], state["keys"].cpu().numpy(),
                         int(state["position"]), int(state["gaussian"]), float(state["cached"])))
