"""Reshardable model/optimizer checkpoints using PyTorch DCP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.optim import Optimizer
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict


def save_distributed_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Collectively save a checkpoint that can be loaded with a new world size."""
    destination = Path(path)
    model_state, optimizer_state = get_state_dict(model, optimizer)
    dcp.save(
        {"model": model_state, "optimizer": optimizer_state},
        storage_writer=dcp.FileSystemWriter(destination, overwrite=True),
        no_dist=not dist.is_initialized(),
    )
    if not dist.is_initialized() or dist.get_rank() == 0:
        (destination / "gopi_metadata.json").write_text(
            json.dumps(metadata or {}, indent=2, default=str) + "\n", encoding="utf-8"
        )
    if dist.is_initialized():
        dist.barrier()
    return destination


def load_distributed_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
) -> dict[str, Any]:
    """Collectively restore model and optimizer state from a DCP directory."""
    source = Path(path)
    if not source.is_dir():
        raise FileNotFoundError(f"distributed checkpoint directory not found: {source}")
    model_state, optimizer_state = get_state_dict(model, optimizer)
    state = {"model": model_state, "optimizer": optimizer_state}
    dcp.load(state, checkpoint_id=source, no_dist=not dist.is_initialized())
    set_state_dict(
        model,
        optimizer,
        model_state_dict=state["model"],
        optim_state_dict=state["optimizer"],
    )
    metadata_path = source / "gopi_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    metadata.setdefault("step", 0)
    metadata.setdefault("trainer", {})
    metadata.setdefault("sampler", {})
    return metadata
