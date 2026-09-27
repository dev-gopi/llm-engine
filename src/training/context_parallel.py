"""Training-side context/sequence parallel collectives."""
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

def reduce_scatter_sequence(tensor: torch.Tensor, *, group=None) -> torch.Tensor:
    """Sum-reduce a full sequence and scatter equal sequence chunks."""
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size(group)==1:return tensor
    world=dist.get_world_size(group)
    if tensor.ndim<2 or tensor.shape[1]%world: raise ValueError("sequence dimension must divide process-group size")
    chunks=list(tensor.chunk(world,dim=1)); out=torch.empty_like(chunks[0])
    # reduce_scatter may be unavailable on some CPU backends; use functional fallback.
    try: dist.reduce_scatter(out,chunks,group=group)
    except RuntimeError:
        full=tensor.clone(); dist.all_reduce(full,group=group); out.copy_(full.chunk(world,dim=1)[dist.get_rank(group)])
    return out

def sequence_parallel_linear(x: torch.Tensor, linear, *, group=None, gather_output: bool=True) -> torch.Tensor:
    """Apply a linear layer to local sequence shards and optionally all-gather output."""
    local=linear(x)
    return gather_sequence(local,group=group) if gather_output else local

def context_parallel_causal_mask(local_query_length:int, total_key_length:int, *, rank:int, world_size:int, device=None) -> torch.Tensor:
    """Boolean causal mask for a contiguous context shard."""
    if min(local_query_length,total_key_length,world_size)<1 or not 0<=rank<world_size: raise ValueError("invalid context-parallel shape")
    offset=rank*local_query_length
    q=torch.arange(local_query_length,device=device)[:,None]+offset; k=torch.arange(total_key_length,device=device)[None,:]
    return k<=q

__all__=["shard_batch_sequence","gather_sequence","reduce_scatter_sequence","sequence_parallel_linear","context_parallel_causal_mask"]
