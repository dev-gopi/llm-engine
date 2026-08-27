"""One-dimensional tensor parallelism for MiniGPT inference."""
from __future__ import annotations

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def validate_tensor_parallel_size(size: int, *, attention_heads: int, kv_heads: int) -> None:
    if size < 1:
        raise ValueError("tensor parallel size must be positive")
    if attention_heads % size or kv_heads % size:
        raise ValueError("attention and KV head counts must be divisible by tensor parallel size")
    if size > 1 and int(os.getenv("WORLD_SIZE", "1")) != size:
        raise RuntimeError(
            f"tensor parallel size {size} requires a torchrun world size of {size}; "
            f"current WORLD_SIZE is {os.getenv('WORLD_SIZE', '1')}"
        )


class VocabParallelLinear(nn.Module):
    """Vocabulary-sharded projection that gathers ordinary full logits."""
    def __init__(self, source: nn.Linear, rank: int, size: int, group=None) -> None:
        super().__init__()
        if source.out_features % size:
            raise ValueError("vocabulary size must be divisible by tensor parallel size")
        width = source.out_features // size
        start, stop = rank * width, (rank + 1) * width
        self.in_features, self.out_features, self.group = source.in_features, source.out_features, group
        self.weight = nn.Parameter(source.weight[start:stop].detach().clone())
        self.bias = nn.Parameter(source.bias[start:stop].detach().clone()) if source.bias is not None else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        local = F.linear(inputs, self.weight, self.bias)
        parts = [torch.empty_like(local) for _ in range(dist.get_world_size(self.group))]
        dist.all_gather(parts, local, group=self.group)
        return torch.cat(parts, dim=-1)


def _rows(source: nn.Linear, rows: torch.Tensor) -> nn.Linear:
    target = nn.Linear(source.in_features, len(rows), bias=source.bias is not None,
                       device=source.weight.device, dtype=source.weight.dtype)
    target.weight.data.copy_(source.weight.data.index_select(0, rows))
    if source.bias is not None:
        target.bias.data.copy_(source.bias.data.index_select(0, rows))
    return target


def _columns(source: nn.Linear, start: int, stop: int, size: int) -> nn.Linear:
    target = nn.Linear(stop - start, source.out_features, bias=source.bias is not None,
                       device=source.weight.device, dtype=source.weight.dtype)
    target.weight.data.copy_(source.weight.data[:, start:stop])
    if source.bias is not None:
        target.bias.data.copy_(source.bias.data / size)
    return target


def parallelize_minigpt(model: nn.Module, *, group=None) -> nn.Module:
    """Shard a fully checkpoint-loaded MiniGPT in-place across the process group."""
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("tensor parallelism requires an initialized torch.distributed process group")
    group = dist.group.WORLD if group is None else group
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    if size == 1:
        return model
    first = model.blocks[0].attn
    validate_tensor_parallel_size(size, attention_heads=first.heads, kv_heads=first.kv_heads)
    for block in model.blocks:
        attn = block.attn
        old_heads, old_kv, head_dim = attn.heads, attn.kv_heads, attn.head_dim
        local_heads, local_kv = old_heads // size, old_kv // size
        qs, qe = rank * local_heads * head_dim, (rank + 1) * local_heads * head_dim
        kv_width = old_kv * head_dim
        ks, ke = rank * local_kv * head_dim, (rank + 1) * local_kv * head_dim
        if attn.qkv_proj is not None:
            dev = attn.qkv_proj.weight.device
            rows = torch.cat((torch.arange(qs, qe, device=dev),
                              torch.arange(attn.dim + ks, attn.dim + ke, device=dev),
                              torch.arange(attn.dim + kv_width + ks, attn.dim + kv_width + ke, device=dev)))
            attn.qkv_proj = _rows(attn.qkv_proj, rows)
        else:
            attn.q_proj = _rows(attn.q_proj, torch.arange(qs, qe, device=attn.q_proj.weight.device))
            attn.k_proj = _rows(attn.k_proj, torch.arange(ks, ke, device=attn.k_proj.weight.device))
            attn.v_proj = _rows(attn.v_proj, torch.arange(ks, ke, device=attn.v_proj.weight.device))
        attn.out_proj = _columns(attn.out_proj, qs, qe, size)
        attn.heads, attn.kv_heads = local_heads, local_kv
        attn.num_kv_groups = local_heads // local_kv
        attn.tensor_parallel_group = group

        ffn = block.ffn
        old_hidden = ffn.hidden_dim
        if old_hidden % size:
            raise ValueError("FFN hidden size must be divisible by tensor parallel size")
        local_hidden = old_hidden // size
        start, stop = rank * local_hidden, (rank + 1) * local_hidden
        dev = ffn.in_proj.weight.device
        rows = (torch.cat((torch.arange(start, stop, device=dev),
                           torch.arange(old_hidden + start, old_hidden + stop, device=dev)))
                if ffn.is_gated else torch.arange(start, stop, device=dev))
        ffn.in_proj = _rows(ffn.in_proj, rows)
        ffn.out_proj = _columns(ffn.out_proj, start, stop, size)
        ffn.hidden_dim, ffn.tensor_parallel_group = local_hidden, group

    model.head = VocabParallelLinear(model.head, rank, size, group)
    model.tensor_parallel_size = size
    return model
