"""FSDP-safe post-training wrapping and sharded checkpoint lifecycle."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import torch
import torch.distributed as dist
from torch import nn
try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, ShardedStateDictConfig, ShardedOptimStateDictConfig
except ImportError:
    FSDP = None

def wrap_post_training(model: nn.Module, *, context, mixed_precision: str = "bf16") -> nn.Module:
    if FSDP is None: raise RuntimeError("PyTorch FSDP is unavailable")
    if context.world_size < 2 or context.device.type != "cuda": raise RuntimeError("FSDP post-training requires multi-rank CUDA")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "none": None}.get(mixed_precision)
    if mixed_precision not in {"none","fp16","bf16"}: raise ValueError("invalid mixed_precision")
    from training.distributed import DistributedTrainer
    return DistributedTrainer.wrap(model, context, strategy="fsdp", mixed_precision=mixed_precision)

def save_sharded_post_training(path: str | Path, model: nn.Module, optimizer, *, metadata: dict[str,Any] | None = None) -> Path:
    if FSDP is None or not isinstance(model, FSDP): raise TypeError("model must be an FSDP instance")
    root=Path(path); root.mkdir(parents=True, exist_ok=True)
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, ShardedStateDictConfig(offload_to_cpu=True), ShardedOptimStateDictConfig(offload_to_cpu=True)):
        state={"model": model.state_dict(), "optimizer": FSDP.optim_state_dict(model, optimizer), "metadata": metadata or {}, "format":"gopi-fsdp-post-v1"}
        torch.save(state, root / f"rank-{dist.get_rank():05d}.pt")
    if dist.get_rank()==0:
        (root/"manifest.json").write_text('{"format":"gopi-fsdp-post-v1"}\n', encoding="utf8")
    dist.barrier()
    return root

def load_sharded_post_training(path: str | Path, model: nn.Module, optimizer, *, map_location="cpu") -> dict:
    if FSDP is None or not isinstance(model, FSDP): raise TypeError("model must be an FSDP instance")
    payload=torch.load(Path(path)/f"rank-{dist.get_rank():05d}.pt", map_location=map_location, weights_only=False)
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, ShardedStateDictConfig(offload_to_cpu=True), ShardedOptimStateDictConfig(offload_to_cpu=True)):
        model.load_state_dict(payload["model"])
        optim_state=FSDP.optim_state_dict_to_load(model, optimizer, payload["optimizer"])
        optimizer.load_state_dict(optim_state)
    return payload


def save_sharded_rlhf_checkpoint(path: str | Path, models: dict[str, nn.Module], optimizers: dict[str, object] | None = None, *, metadata: dict[str,Any] | None = None) -> Path:
    """Save policy/old-policy/reference/reward/value FSDP modules and optimizer state per rank."""
    if FSDP is None: raise RuntimeError("PyTorch FSDP is unavailable")
    if not models or any(not isinstance(m,FSDP) for m in models.values()): raise TypeError("all RLHF models must be FSDP instances")
    root=Path(path); root.mkdir(parents=True,exist_ok=True); optimizers=optimizers or {}
    payload={"format":"gopi-fsdp-rlhf-v1","metadata":metadata or {},"models":{},"optimizers":{}}
    cfg=ShardedStateDictConfig(offload_to_cpu=True); ocfg=ShardedOptimStateDictConfig(offload_to_cpu=True)
    for name,model in models.items():
        with FSDP.state_dict_type(model,StateDictType.SHARDED_STATE_DICT,cfg,ocfg):
            payload["models"][name]=model.state_dict()
            if name in optimizers: payload["optimizers"][name]=FSDP.optim_state_dict(model,optimizers[name])
    torch.save(payload,root/f"rank-{dist.get_rank():05d}.pt")
    dist.barrier()
    return root

def load_sharded_rlhf_checkpoint(path: str | Path, models: dict[str, nn.Module], optimizers: dict[str, object] | None = None, *, map_location="cpu") -> dict:
    if FSDP is None: raise RuntimeError("PyTorch FSDP is unavailable")
    payload=torch.load(Path(path)/f"rank-{dist.get_rank():05d}.pt",map_location=map_location,weights_only=False); optimizers=optimizers or {}
    if payload.get("format")!="gopi-fsdp-rlhf-v1": raise ValueError("unsupported RLHF checkpoint")
    cfg=ShardedStateDictConfig(offload_to_cpu=True); ocfg=ShardedOptimStateDictConfig(offload_to_cpu=True)
    for name,model in models.items():
        if name not in payload["models"]: raise KeyError(f"checkpoint missing model {name!r}")
        with FSDP.state_dict_type(model,StateDictType.SHARDED_STATE_DICT,cfg,ocfg):
            model.load_state_dict(payload["models"][name])
            if name in optimizers and name in payload["optimizers"]:
                optimizers[name].load_state_dict(FSDP.optim_state_dict_to_load(model,optimizers[name],payload["optimizers"][name]))
    return payload
