"""Training-side context/sequence parallel helpers."""
from __future__ import annotations
import torch
import torch.distributed as dist

def shard_batch_sequence(batch: dict[str,torch.Tensor], *, rank:int, world_size:int, keys=("input_ids","attention_mask","labels")):
    out=dict(batch)
    for key in keys:
        value=out.get(key)
        if value is None: continue
        if value.ndim<2 or value.shape[1]%world_size: raise ValueError(f"{key} sequence length is not divisible by world_size")
        width=value.shape[1]//world_size; out[key]=value[:,rank*width:(rank+1)*width].contiguous()
    return out

def gather_sequence(tensor: torch.Tensor, group=None):
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size(group)==1:return tensor
    parts=[torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]; dist.all_gather(parts,tensor,group=group); return torch.cat(parts,dim=1)
