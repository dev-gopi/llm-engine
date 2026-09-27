"""First-class distributed parallel topology construction and model wiring."""

from __future__ import annotations

from typing import Any, Mapping

import torch.distributed as dist
from torch import nn

from inference.tensor_parallel import parallelize_minigpt
from training.parallelism import ParallelDegrees, ParallelMesh

_ALIASES = {
    "data": ("data", "data_parallel", "dp"),
    "tensor": ("tensor", "tensor_parallel", "tp"),
    "pipeline": ("pipeline", "pipeline_parallel", "pp"),
    "expert": ("expert", "expert_parallel", "ep"),
    "context": ("context", "context_parallel", "cp"),
    "sequence": ("sequence", "sequence_parallel", "sp"),
}


def _value(mapping: Mapping[str, Any], name: str, default: int = 1) -> int:
    for key in _ALIASES[name]:
        if key in mapping:
            value = mapping[key]
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
            return value
    return default


def resolve_parallel_degrees(
    config: Mapping[str, Any], *, world_size: int
) -> ParallelDegrees:
    """Resolve DP/TP/PP/EP/CP/SP degrees and infer DP when omitted.

    The accepted config is either ``parallelism: {...}`` or the same keys at the
    training-config top level.  Non-data degrees are explicit; data parallelism
    defaults to the remaining world-size factor.
    """
    if world_size < 1:
        raise ValueError("world_size must be positive")
    nested = config.get("parallelism", {})
    if nested is None:
        nested = {}
    if not isinstance(nested, Mapping):
        raise ValueError("parallelism must be a mapping")
    merged = dict(config)
    merged.update(nested)
    tensor = _value(merged, "tensor")
    pipeline = _value(merged, "pipeline")
    expert = _value(merged, "expert")
    context = _value(merged, "context")
    sequence = _value(merged, "sequence")
    non_data = tensor * pipeline * expert * context * sequence
    explicit_data = any(key in merged for key in _ALIASES["data"])
    if explicit_data:
        data = _value(merged, "data")
    else:
        if world_size % non_data:
            raise ValueError(
                f"WORLD_SIZE={world_size} is not divisible by non-data parallel degree {non_data}"
            )
        data = world_size // non_data
    degrees = ParallelDegrees(data, tensor, pipeline, expert, context, sequence)
    if degrees.world_size != world_size:
        raise ValueError(
            f"parallel degrees multiply to {degrees.world_size}, but WORLD_SIZE is {world_size}"
        )
    return degrees


def build_parallel_mesh(
    config: Mapping[str, Any], *, world_size: int
) -> ParallelMesh | None:
    degrees = resolve_parallel_degrees(config, world_size=world_size)
    if world_size == 1:
        return None
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "parallel mesh requires an initialized distributed process group"
        )
    return ParallelMesh(degrees)


def apply_model_parallelism(model: nn.Module, mesh: ParallelMesh | None) -> nn.Module:
    """Wire model components that already have production distributed adapters.

    Expert parallelism is applied immediately.  The mesh is attached to the
    model so attention/block implementations and checkpoint code can consume
    CP/SP/PP/TP groups without reconstructing topology independently.
    """
    setattr(model, "parallel_mesh", mesh)
    if mesh is None:
        return model
    if mesh.degree("tensor") > 1:
        parallelize_minigpt(model, group=mesh.group("tensor"))
    if mesh.degree("expert") > 1:
        expert_group = mesh.group("expert")
        found_moe = False
        for block in getattr(model, "blocks", []):
            if hasattr(getattr(block, "ffn", None), "expert_parallel_group"):
                block.ffn.expert_parallel_group = expert_group
                found_moe = True
        if not found_moe:
            raise ValueError(
                "expert_parallel > 1 requires a model with SparseMoE feed-forward layers"
            )
        setattr(model, "expert_parallel_size", mesh.degree("expert"))
    context_group = mesh.group("context")
    sequence_group = mesh.group("sequence")
    setattr(model, "context_parallel_group", context_group)
    setattr(model, "context_parallel_rank", mesh.local_rank("context"))
    setattr(model, "context_parallel_size", mesh.degree("context"))
    setattr(model, "sequence_parallel_group", sequence_group)
    setattr(model, "pipeline_parallel_group", mesh.group("pipeline"))
    setattr(model, "tensor_parallel_group", mesh.group("tensor"))
    for block in getattr(model, "blocks", []):
        if hasattr(block, "attn"):
            block.attn.context_parallel_group = context_group
        block.sequence_parallel_group = sequence_group
    return model


__all__ = ["resolve_parallel_degrees", "build_parallel_mesh", "apply_model_parallelism"]
