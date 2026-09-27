"""Context/sequence/pipeline/expert parallel runtime primitives.

These components are deliberately composable with the existing tensor-parallel contract.
They do not change single-device model behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor, nn


@dataclass(frozen=True)
class ParallelRuntimeConfig:
    context_parallel: int = 1
    sequence_parallel: int = 1
    pipeline_parallel: int = 1
    expert_parallel: int = 1

    def world_size(self):
        return (
            self.context_parallel
            * self.sequence_parallel
            * self.pipeline_parallel
            * self.expert_parallel
        )

    def validate(self, *, world_size: int, sequence_length: int, num_experts: int = 1):
        for name, v in (
            ("context_parallel", self.context_parallel),
            ("sequence_parallel", self.sequence_parallel),
            ("pipeline_parallel", self.pipeline_parallel),
            ("expert_parallel", self.expert_parallel),
        ):
            if not isinstance(v, int) or v < 1:
                raise ValueError(f"{name} must be positive")
        if world_size != self.world_size():
            raise ValueError(f"parallel world size {world_size} != {self.world_size()}")
        if sequence_length % self.context_parallel:
            raise ValueError("sequence length must divide context parallel degree")
        if sequence_length % self.sequence_parallel:
            raise ValueError("sequence length must divide sequence parallel degree")
        if num_experts % self.expert_parallel:
            raise ValueError("number of experts must divide expert parallel degree")


def sequence_shard(hidden_states: Tensor, rank: int, world_size: int) -> Tensor:
    if hidden_states.ndim < 2 or hidden_states.shape[1] % world_size:
        raise ValueError("sequence dimension must divide world size")
    width = hidden_states.shape[1] // world_size
    return hidden_states[:, rank * width : (rank + 1) * width].contiguous()


def sequence_gather(local: Tensor, group=None) -> Tensor:
    if (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_world_size(group) == 1
    ):
        return local
    parts = [torch.empty_like(local) for _ in range(dist.get_world_size(group))]
    dist.all_gather(parts, local, group=group)
    return torch.cat(parts, dim=1)


class PipelineStage(nn.Module):
    """A stage containing a contiguous subset of Transformer blocks."""

    def __init__(
        self, blocks: list[nn.Module], *, first: bool = False, last: bool = False
    ):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.first = first
        self.last = last

    def forward(self, hidden_states: Tensor, **kwargs):
        for block in self.blocks:
            hidden_states = block(hidden_states, **kwargs)
        return hidden_states


def split_pipeline_blocks(blocks: nn.ModuleList, stages: int) -> list[list[nn.Module]]:
    if stages < 1 or len(blocks) < stages:
        raise ValueError("pipeline stages must be positive and not exceed block count")
    q, r = divmod(len(blocks), stages)
    out = []
    start = 0
    for i in range(stages):
        n = q + (1 if i < r else 0)
        out.append(list(blocks[start : start + n]))
        start += n
    return out


class ExpertParallelShard:
    """Checkpoint/runtime helper that owns a deterministic subset of MoE experts."""

    def __init__(self, num_experts: int, rank: int, world_size: int):
        if (
            num_experts < 1
            or world_size < 1
            or not 0 <= rank < world_size
            or num_experts % world_size
        ):
            raise ValueError("experts/world size must be compatible")
        self.num_experts, self.rank, self.world_size = num_experts, rank, world_size
        self.indices = tuple(range(rank, num_experts, world_size))

    def shard_state_dict(
        self, state: dict[str, Tensor], prefix: str = "experts"
    ) -> dict[str, Tensor]:
        allowed = {f"{prefix}.{i}." for i in self.indices}
        return {k: v for k, v in state.items() if any(k.startswith(p) for p in allowed)}

    def local_indices(self):
        return self.indices


def load_expert_shard(
    module: nn.Module,
    full_state: dict[str, Tensor],
    shard: ExpertParallelShard,
    *,
    strict: bool = False,
):
    local = shard.shard_state_dict(full_state)
    missing = module.load_state_dict(local, strict=False)
    if strict and (missing.missing_keys or missing.unexpected_keys):
        raise RuntimeError(f"expert shard load mismatch: {missing}")
    return missing


class MoETensorParallelAdapter(nn.Module):
    """Memory-sharded MoE adapter: replicated router, local experts, all-reduce output."""

    def __init__(self, moe: nn.Module, rank: int, world_size: int, group=None):
        super().__init__()
        if not hasattr(moe, "experts") or not hasattr(moe, "router"):
            raise TypeError("expected SparseMoE-compatible module")
        if moe.num_experts % world_size:
            raise ValueError("num_experts must divide tensor parallel world size")
        self.router = moe.router
        self.num_experts = moe.num_experts
        self.experts_per_token = moe.experts_per_token
        self.dim = moe.dim
        self.group = group
        self.global_expert_indices = tuple(range(rank, moe.num_experts, world_size))
        self.experts = nn.ModuleList(
            [moe.experts[i] for i in self.global_expert_indices]
        )
        self.router_jitter = moe.router_jitter
        self.capacity_factor = moe.capacity_factor
        self.min_capacity = moe.min_capacity

    def forward(self, hidden_states: Tensor) -> Tensor:
        original = hidden_states.shape
        tokens = hidden_states.reshape(-1, self.dim)
        logits = self.router(tokens)
        probs = torch.softmax(logits.float(), dim=-1)
        top_logits, top_experts = torch.topk(logits, self.experts_per_token, dim=-1)
        weights = torch.softmax(top_logits.float(), dim=-1).to(tokens.dtype)
        output = torch.zeros_like(tokens)
        local_map = {
            global_i: i for i, global_i in enumerate(self.global_expert_indices)
        }
        for global_i, local_i in local_map.items():
            ti, ri = torch.where(top_experts == global_i)
            if ti.numel():
                output.index_add_(
                    0,
                    ti,
                    self.experts[local_i](tokens.index_select(0, ti))
                    * weights[ti, ri].unsqueeze(-1),
                )
        if (
            dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size(self.group) > 1
        ):
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.group)
        return output.reshape(original)


def parallelize_moe_tensor_parallel(model: nn.Module, *, group=None) -> nn.Module:
    """Replace every SparseMoE FFN with a memory-sharded expert-parallel adapter."""
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "MoE tensor parallelism requires an initialized process group"
        )
    group = dist.group.WORLD if group is None else group
    size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    for block in getattr(model, "blocks", []):
        if hasattr(block.ffn, "experts"):
            block.ffn = MoETensorParallelAdapter(block.ffn, rank, size, group)
    model.moe_tensor_parallel_size = size
    return model
