"""Small, explicit helpers for torch.distributed training."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
except ImportError:  # pragma: no cover - depends on the installed PyTorch build
    FullyShardedDataParallel = None


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


class DistributedTrainer:
    @staticmethod
    def initialize(backend: str | None = None) -> DistributedContext:
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        rank = int(os.getenv("RANK", "0"))
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
            raise ValueError("WORLD_SIZE, RANK, and LOCAL_RANK are inconsistent")
        if world_size > 1 and not dist.is_initialized():
            resolved_backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
            dist.init_process_group(backend=resolved_backend, init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        return DistributedContext(rank, local_rank, world_size, device)

    @staticmethod
    def wrap(
        model: nn.Module,
        context: DistributedContext,
        *,
        strategy: str = "ddp",
        mixed_precision: str = "none",
    ) -> nn.Module:
        strategy = strategy.lower()
        if strategy not in {"none", "ddp", "fsdp", "fsdp_hybrid"}:
            raise ValueError("distributed_strategy must be none, ddp, fsdp, or fsdp_hybrid")
        model = model.to(context.device)
        if context.world_size > 1 and strategy == "none":
            raise ValueError(
                "distributed_strategy='none' cannot be used with WORLD_SIZE > 1"
            )
        if context.world_size == 1 or strategy == "none":
            return model
        if strategy.startswith("fsdp"):
            if FullyShardedDataParallel is None:
                raise RuntimeError("this PyTorch build does not provide FSDP")
            if context.device.type != "cuda":
                raise RuntimeError("FSDP training requires CUDA in this engine")
            from functools import partial
            from model.transformer_block import TransformerBlock

            dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(mixed_precision)
            precision = None if dtype is None else MixedPrecision(
                param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype
            )
            sharding = (
                ShardingStrategy.HYBRID_SHARD
                if strategy == "fsdp_hybrid"
                else ShardingStrategy.FULL_SHARD
            )
            return FullyShardedDataParallel(
                model,
                auto_wrap_policy=partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls={TransformerBlock},
                ),
                sharding_strategy=sharding,
                mixed_precision=precision,
                device_id=context.device,
                use_orig_params=True,
                limit_all_gathers=True,
            )
        device_ids = [context.local_rank] if context.device.type == "cuda" else None
        return DistributedDataParallel(model, device_ids=device_ids)

    @staticmethod
    def mean(value: Tensor, context: DistributedContext) -> Tensor:
        if context.world_size == 1:
            return value
        reduced = value.detach().clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        return reduced / context.world_size

    @staticmethod
    def barrier(context: DistributedContext) -> None:
        if context.world_size > 1:
            dist.barrier()

    @staticmethod
    def shutdown() -> None:
        if dist.is_initialized():
            dist.destroy_process_group()
