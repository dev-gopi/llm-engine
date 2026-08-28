"""Reshardable model/optimizer checkpoints using PyTorch DCP."""

from __future__ import annotations

import json
import hashlib
import os
import random
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
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
    scheduler: Any = None,
    scaler: Any = None,
) -> Path:
    """Collectively save sharded weights and rank-local resumable runtime state."""
    destination = Path(path)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    transaction = [uuid.uuid4().hex if rank == 0 else ""]
    if dist.is_initialized():
        dist.broadcast_object_list(transaction, src=0)
    staging = destination.with_name(f".{destination.name}.staging-{transaction[0]}")
    model_state, optimizer_state = get_state_dict(model, optimizer)
    dcp.save(
        {"model": model_state, "optimizer": optimizer_state},
        storage_writer=dcp.FileSystemWriter(staging, overwrite=False),
        no_dist=not dist.is_initialized(),
    )
    rank_state = {
        "format_version": 1,
        "rank": rank,
        "world_size": world_size,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": _numpy_rng_state(),
    }
    _atomic_torch_save(rank_state, staging / f"gopi_rank_{rank:05d}.pt")
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        saved_metadata = dict(metadata or {})
        saved_metadata["checkpoint_world_size"] = world_size
        (staging / "gopi_metadata.json").write_text(
            json.dumps(saved_metadata, indent=2, default=str) + "\n", encoding="utf-8"
        )
        manifest = _checksum_manifest(staging)
        (staging / "gopi_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (staging / "gopi_complete").write_text(transaction[0] + "\n", encoding="utf-8")
        backup = destination.with_name(f".{destination.name}.previous-{transaction[0]}")
        if destination.exists():
            os.replace(destination, backup)
        try:
            os.replace(staging, destination)
        except BaseException:
            if backup.exists() and not destination.exists():
                os.replace(backup, destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    if dist.is_initialized():
        dist.barrier()
    return destination


def load_distributed_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    *,
    scheduler: Any = None,
    scaler: Any = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore sharded weights plus compatible scheduler, scaler, and RNG state."""
    source = Path(path)
    if not source.is_dir():
        raise FileNotFoundError(f"distributed checkpoint directory not found: {source}")
    _verify_checkpoint(source)
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
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    saved_world_size = int(metadata.get("checkpoint_world_size", world_size))
    state_rank = rank if saved_world_size == world_size else 0
    rank_state_path = source / f"gopi_rank_{state_rank:05d}.pt"
    runtime_state_restored = False
    rng_restored = False
    if rank_state_path.is_file():
        rank_state = torch.load(rank_state_path, map_location="cpu", weights_only=True)
        if scheduler is not None and rank_state.get("scheduler") is not None:
            scheduler.load_state_dict(rank_state["scheduler"])
        if scaler is not None and rank_state.get("scaler") is not None:
            scaler.load_state_dict(rank_state["scaler"])
        runtime_state_restored = True
        if restore_rng and saved_world_size == world_size:
            _restore_rng_state(rank_state)
            rng_restored = True
    metadata.setdefault("step", 0)
    metadata.setdefault("trainer", {})
    metadata.setdefault("sampler", {})
    metadata["runtime_state_restored"] = runtime_state_restored
    metadata["rng_restored"] = rng_restored
    return metadata


def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _checksum_manifest(directory: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(candidate for candidate in directory.rglob("*") if candidate.is_file()):
        relative = path.relative_to(directory).as_posix()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        files[relative] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}
    return {"format": "gopi-distributed-checkpoint-v2", "files": files}


def _verify_checkpoint(directory: Path) -> None:
    manifest_path = directory / "gopi_manifest.json"
    if not (directory / "gopi_complete").is_file() or not manifest_path.is_file():
        raise ValueError("distributed checkpoint is incomplete or uses an unsupported legacy format")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "gopi-distributed-checkpoint-v2":
        raise ValueError("unsupported distributed checkpoint manifest")
    for relative, expected in manifest.get("files", {}).items():
        path = directory / relative
        if not path.is_file() or path.stat().st_size != int(expected["bytes"]):
            raise ValueError(f"distributed checkpoint file is missing or truncated: {relative}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected["sha256"]:
            raise ValueError(f"distributed checkpoint checksum mismatch: {relative}")


def _numpy_rng_state() -> dict[str, Any]:
    name, keys, position, gaussian, cached = np.random.get_state()
    return {
        "name": name,
        "keys": torch.from_numpy(keys.copy()),
        "position": position,
        "gaussian": gaussian,
        "cached": cached,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch_rng_state"].cpu().to(torch.uint8))
    if torch.cuda.is_available() and state.get("cuda_rng_state") is not None:
        cuda_states = [value.cpu().to(torch.uint8) for value in state["cuda_rng_state"]]
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)
    random.setstate(state["python_rng_state"])
    numpy_state = state["numpy_rng_state"]
    np.random.set_state((
        numpy_state["name"],
        numpy_state["keys"].cpu().numpy(),
        int(numpy_state["position"]),
        int(numpy_state["gaussian"]),
        float(numpy_state["cached"]),
    ))
