"""Small, explicit helpers for torch.distributed training."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel


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
    def wrap(model: nn.Module, context: DistributedContext) -> nn.Module:
        model = model.to(context.device)
        if context.world_size == 1:
            return model
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
