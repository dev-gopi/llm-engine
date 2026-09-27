"""FSDP-safe post-training wrapping and sharded checkpoint lifecycle.

The helpers in this module deliberately keep the post-training scripts small and
preserve the existing single-file checkpoint format for non-FSDP runs. FSDP
runs use one shard per rank plus a compact manifest; optimizer state is restored
with PyTorch's FSDP optimizer-state conversion APIs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

try:
    from torch.distributed.fsdp import (
        FullOptimStateDictConfig,
        FullStateDictConfig,
        ShardedOptimStateDictConfig,
        ShardedStateDictConfig,
        StateDictType,
    )
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
    )
except ImportError:  # pragma: no cover - depends on installed PyTorch
    FSDP = None


FORMAT = "gopi-fsdp-post-v2"
RLHF_FORMAT = "gopi-fsdp-rlhf-v2"


def read_fsdp_manifest(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    manifest = root / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"FSDP checkpoint manifest not found: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf8"))
    if payload.get("format") not in {
        FORMAT,
        RLHF_FORMAT,
        "gopi-fsdp-post-v1",
        "gopi-fsdp-rlhf-v1",
    }:
        raise ValueError("unsupported FSDP checkpoint manifest")
    return payload


def is_fsdp_checkpoint(path: str | Path) -> bool:
    root = Path(path)
    manifest = root / "manifest.json"
    if not root.is_dir() or not manifest.is_file():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("format") in {
        FORMAT,
        RLHF_FORMAT,
        "gopi-fsdp-post-v1",
        "gopi-fsdp-rlhf-v1",
    }


def _require_fsdp(model: nn.Module) -> None:
    if FSDP is None:
        raise RuntimeError("PyTorch FSDP is unavailable")
    if not isinstance(model, FSDP):
        raise TypeError("model must be an FSDP instance")
    if not dist.is_initialized():
        raise RuntimeError("FSDP checkpointing requires an initialized process group")


def is_fsdp_model(model: nn.Module) -> bool:
    """Return True when ``model`` is a live FSDP wrapper."""
    return FSDP is not None and isinstance(model, FSDP)


def copy_fsdp_weights_to_model(source: nn.Module, target: nn.Module) -> None:
    """Collectively materialize FSDP weights and copy them into a replica.

    Online RLHF rollouts intentionally run on ordinary replica models so token-by-token
    generation does not repeatedly enter FSDP all-gathers.  Every rank participates in
    the full-state materialization and receives the same CPU state dict.
    """
    _require_fsdp(source)
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
    with FSDP.state_dict_type(source, StateDictType.FULL_STATE_DICT, cfg):
        state = source.state_dict()
    target.load_state_dict(state, strict=True)


def wrap_post_training(
    model: nn.Module, *, context, mixed_precision: str = "bf16", strategy: str = "fsdp"
) -> nn.Module:
    if FSDP is None:
        raise RuntimeError("PyTorch FSDP is unavailable")
    if context.world_size < 2 or context.device.type != "cuda":
        raise RuntimeError("FSDP post-training requires multi-rank CUDA")
    if strategy not in {"fsdp", "fsdp_hybrid"}:
        raise ValueError("FSDP post-training strategy must be fsdp or fsdp_hybrid")
    if mixed_precision not in {"none", "fp16", "bf16"}:
        raise ValueError("invalid mixed_precision")
    from training.distributed import DistributedTrainer

    return DistributedTrainer.wrap(
        model, context, strategy=strategy, mixed_precision=mixed_precision
    )


def save_sharded_post_training(
    path: str | Path,
    model: nn.Module,
    optimizer,
    *,
    metadata: dict[str, Any] | None = None,
    trainer_state: dict[str, Any] | None = None,
    scheduler=None,
    scaler=None,
) -> Path:
    """Save an FSDP policy/reward-model checkpoint as rank-local shards."""
    _require_fsdp(model)
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    state_cfg = ShardedStateDictConfig(offload_to_cpu=True)
    optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)
    with FSDP.state_dict_type(
        model, StateDictType.SHARDED_STATE_DICT, state_cfg, optim_cfg
    ):
        payload = {
            "format": FORMAT,
            "model": model.state_dict(),
            "optimizer": FSDP.optim_state_dict(model, optimizer),
            "metadata": metadata or {},
            "trainer": trainer_state or {},
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
        }
        torch.save(payload, root / f"rank-{dist.get_rank():05d}.pt")
    # Keep an elastic recovery payload alongside the normal fast shards.  It is
    # only materialized on rank 0 and is used when the resume world size changes.
    full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    full_optim_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, full_cfg, full_optim_cfg
    ):
        full_model = model.state_dict()
        full_optimizer = FSDP.optim_state_dict(model, optimizer)
    if dist.get_rank() == 0:
        torch.save(
            {
                "format": FORMAT,
                "model": full_model,
                "optimizer": full_optimizer,
                "metadata": metadata or {},
                "trainer": trainer_state or {},
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict() if scaler is not None else None,
            },
            root / "elastic-full.pt",
        )
    if dist.get_rank() == 0:
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "format": FORMAT,
                    "world_size": dist.get_world_size(),
                    "metadata": metadata or {},
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf8",
        )
    dist.barrier()
    return root


def load_sharded_post_training(
    path: str | Path,
    model: nn.Module,
    optimizer,
    *,
    scheduler=None,
    scaler=None,
    map_location="cpu",
) -> dict:
    """Restore an exact FSDP post-training checkpoint on every rank."""
    _require_fsdp(model)
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf8"))
    saved_world_size = manifest.get("world_size")
    world_size_changed = (
        saved_world_size is not None and int(saved_world_size) != dist.get_world_size()
    )
    if world_size_changed:
        elastic_path = root / "elastic-full.pt"
        if not elastic_path.is_file():
            raise ValueError(
                f"FSDP checkpoint world_size={saved_world_size} differs from current world_size={dist.get_world_size()} "
                "and this legacy checkpoint has no elastic full-state payload"
            )
        payload = (
            torch.load(elastic_path, map_location=map_location, weights_only=False)
            if dist.get_rank() == 0
            else {"format": FORMAT}
        )
        full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg):
            model.load_state_dict(payload.get("model", {}), strict=True)
        full_optim = payload.get("optimizer") if dist.get_rank() == 0 else None
        sharded_optim = FSDP.scatter_full_optim_state_dict(
            full_optim, model, optim=optimizer
        )
        optimizer.load_state_dict(sharded_optim)
        metadata_payload = [payload if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(metadata_payload, src=0)
        payload = metadata_payload[0]
        if scheduler is not None and payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
        if scaler is not None and payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])
        return payload
    payload = torch.load(
        root / f"rank-{dist.get_rank():05d}.pt",
        map_location=map_location,
        weights_only=False,
    )
    if payload.get("format") not in {FORMAT, "gopi-fsdp-post-v1"}:
        raise ValueError("unsupported FSDP post-training checkpoint format")
    state_cfg = ShardedStateDictConfig(offload_to_cpu=True)
    optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)
    with FSDP.state_dict_type(
        model, StateDictType.SHARDED_STATE_DICT, state_cfg, optim_cfg
    ):
        model.load_state_dict(payload["model"])
        optim_state = FSDP.optim_state_dict_to_load(
            model, optimizer, payload["optimizer"]
        )
        optimizer.load_state_dict(optim_state)
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    return payload


def save_sharded_rlhf_checkpoint(
    path: str | Path,
    models: dict[str, nn.Module],
    optimizers: dict[str, object] | None = None,
    *,
    metadata: dict[str, Any] | None = None,
    extra_state: dict[str, Any] | None = None,
) -> Path:
    """Save multiple FSDP RLHF modules and optional optimizer state per rank."""
    if FSDP is None:
        raise RuntimeError("PyTorch FSDP is unavailable")
    if not models or any(not isinstance(m, FSDP) for m in models.values()):
        raise TypeError("all RLHF models must be FSDP instances")
    if not dist.is_initialized():
        raise RuntimeError("FSDP checkpointing requires an initialized process group")
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    optimizers = optimizers or {}
    payload = {
        "format": RLHF_FORMAT,
        "metadata": metadata or {},
        "models": {},
        "optimizers": {},
        "extra_state": extra_state or {},
    }
    cfg = ShardedStateDictConfig(offload_to_cpu=True)
    ocfg = ShardedOptimStateDictConfig(offload_to_cpu=True)
    for name, model in models.items():
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, cfg, ocfg):
            payload["models"][name] = model.state_dict()
            if name in optimizers:
                payload["optimizers"][name] = FSDP.optim_state_dict(
                    model, optimizers[name]
                )
    torch.save(payload, root / f"rank-{dist.get_rank():05d}.pt")
    elastic_payload = {
        "format": RLHF_FORMAT,
        "metadata": metadata or {},
        "models": {},
        "optimizers": {},
        "extra_state": extra_state or {},
    }
    full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    full_optim_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
    for name, model in models.items():
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, full_cfg, full_optim_cfg
        ):
            elastic_payload["models"][name] = model.state_dict()
            if name in optimizers:
                elastic_payload["optimizers"][name] = FSDP.optim_state_dict(
                    model, optimizers[name]
                )
    if dist.get_rank() == 0:
        torch.save(elastic_payload, root / "elastic-full.pt")
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "format": RLHF_FORMAT,
                    "world_size": dist.get_world_size(),
                    "metadata": metadata or {},
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf8",
        )
    dist.barrier()
    return root


def load_sharded_rlhf_checkpoint(
    path: str | Path,
    models: dict[str, nn.Module],
    optimizers: dict[str, object] | None = None,
    *,
    map_location="cpu",
) -> dict:
    if FSDP is None:
        raise RuntimeError("PyTorch FSDP is unavailable")
    if not dist.is_initialized():
        raise RuntimeError("FSDP checkpointing requires an initialized process group")
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf8"))
    saved_world_size = manifest.get("world_size")
    world_size_changed = (
        saved_world_size is not None and int(saved_world_size) != dist.get_world_size()
    )
    if world_size_changed:
        elastic_path = root / "elastic-full.pt"
        if not elastic_path.is_file():
            raise ValueError(
                f"FSDP checkpoint world_size={saved_world_size} differs from current world_size={dist.get_world_size()} "
                "and this legacy checkpoint has no elastic full-state payload"
            )
        full_payload = (
            torch.load(elastic_path, map_location=map_location, weights_only=False)
            if dist.get_rank() == 0
            else None
        )
        header = None
        if dist.get_rank() == 0:
            if full_payload.get("format") not in {RLHF_FORMAT, "gopi-fsdp-rlhf-v1"}:
                raise ValueError("unsupported RLHF checkpoint")
            header = {
                "format": full_payload.get("format", RLHF_FORMAT),
                "metadata": full_payload.get("metadata", {}),
                "extra_state": full_payload.get("extra_state", {}),
                "model_names": tuple(full_payload.get("models", {})),
                "optimizer_names": tuple(full_payload.get("optimizers", {})),
            }
        header_list = [header]
        dist.broadcast_object_list(header_list, src=0)
        header = header_list[0]
        optimizers = optimizers or {}
        full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        for name, model in models.items():
            if name not in header["model_names"]:
                raise KeyError(f"checkpoint missing model {name!r}")
            rank_state = full_payload["models"][name] if dist.get_rank() == 0 else {}
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg):
                model.load_state_dict(rank_state, strict=True)
            if name in optimizers and name in header["optimizer_names"]:
                full_optim = (
                    full_payload["optimizers"][name] if dist.get_rank() == 0 else None
                )
                optimizers[name].load_state_dict(
                    FSDP.scatter_full_optim_state_dict(
                        full_optim, model, optim=optimizers[name]
                    )
                )
        # Do not broadcast full model/optimizer tensors as Python objects.  The
        # FSDP collectives above distribute them directly to the new topology.
        return {
            "format": header["format"],
            "metadata": header["metadata"],
            "extra_state": header["extra_state"],
            "models": {name: None for name in header["model_names"]},
            "optimizers": {name: None for name in header["optimizer_names"]},
        }
    payload = torch.load(
        root / f"rank-{dist.get_rank():05d}.pt",
        map_location=map_location,
        weights_only=False,
    )
    optimizers = optimizers or {}
    if payload.get("format") not in {RLHF_FORMAT, "gopi-fsdp-rlhf-v1"}:
        raise ValueError("unsupported RLHF checkpoint")
    cfg = ShardedStateDictConfig(offload_to_cpu=True)
    ocfg = ShardedOptimStateDictConfig(offload_to_cpu=True)
    for name, model in models.items():
        if name not in payload["models"]:
            raise KeyError(f"checkpoint missing model {name!r}")
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, cfg, ocfg):
            model.load_state_dict(payload["models"][name])
            if name in optimizers and name in payload["optimizers"]:
                optimizers[name].load_state_dict(
                    FSDP.optim_state_dict_to_load(
                        model, optimizers[name], payload["optimizers"][name]
                    )
                )
    return payload
