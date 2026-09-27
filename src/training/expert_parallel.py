"""Expert-parallel checkpoint and routing utilities."""
from __future__ import annotations
from pathlib import Path
import torch

def expert_indices(num_experts:int, rank:int, world_size:int)->tuple[int,...]:
    if num_experts<1 or world_size<1 or num_experts%world_size or not 0<=rank<world_size: raise ValueError("incompatible expert topology")
    return tuple(range(rank,num_experts,world_size))

def save_expert_shard(path, state_dict, *, num_experts:int, rank:int, world_size:int):
    idx=expert_indices(num_experts,rank,world_size); prefixes=tuple(f"experts.{i}." for i in idx)
    shard={k:v for k,v in state_dict.items() if k.startswith(prefixes)}
    dst=Path(path); dst.parent.mkdir(parents=True,exist_ok=True); torch.save({"format":"gopi-expert-v1","rank":rank,"world_size":world_size,"indices":idx,"state_dict":shard},dst); return dst

def load_expert_shard(path, module, *, strict=False):
    payload=torch.load(path,map_location="cpu",weights_only=False)
    if payload.get("format")!="gopi-expert-v1": raise ValueError("unsupported expert shard format")
    result=module.load_state_dict(payload["state_dict"],strict=False)
    if strict and (result.missing_keys or result.unexpected_keys): raise RuntimeError(str(result))
    return payload
