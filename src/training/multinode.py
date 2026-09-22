"""Preflight checks and collective smoke validation for torchrun jobs."""
from __future__ import annotations

import os
import socket
from dataclasses import asdict, dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedTopology:
    world_size: int
    rank: int
    local_rank: int
    local_world_size: int
    node_rank: int
    nodes: int
    master_addr: str
    master_port: int

    def to_dict(self) -> dict:
        return asdict(self)


def topology_from_environment(environment: dict[str, str] | None = None) -> DistributedTopology:
    env = os.environ if environment is None else environment
    required = ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
    missing = [key for key in required if key not in env]
    if missing:
        raise ValueError(f"missing torchrun environment variables: {', '.join(missing)}")
    world, rank = int(env["WORLD_SIZE"]), int(env["RANK"])
    local_rank, local_world = int(env["LOCAL_RANK"]), int(env["LOCAL_WORLD_SIZE"])
    port = int(env["MASTER_PORT"])
    if world < 1 or local_world < 1 or world % local_world:
        raise ValueError("WORLD_SIZE must be positive and divisible by LOCAL_WORLD_SIZE")
    if not 0 <= rank < world or not 0 <= local_rank < local_world:
        raise ValueError("global or local rank is outside its topology")
    if not 1 <= port <= 65535:
        raise ValueError("MASTER_PORT must be in [1, 65535]")
    node_rank = rank // local_world
    return DistributedTopology(world, rank, local_rank, local_world, node_rank,
                               world // local_world, env["MASTER_ADDR"], port)


def validate_fsdp_topology(
    topology: DistributedTopology, *, minimum_world_size: int = 2,
    required_nodes: int | None = None,
) -> None:
    """Reject FSDP launches that cannot meet their declared distributed scope."""
    if minimum_world_size < 2:
        raise ValueError("minimum_world_size must be at least two for FSDP")
    if topology.world_size < minimum_world_size:
        raise ValueError(
            f"FSDP requires at least {minimum_world_size} ranks, got {topology.world_size}"
        )
    if required_nodes is not None:
        if required_nodes < 1:
            raise ValueError("required_nodes must be positive")
        if topology.nodes != required_nodes:
            raise ValueError(
                f"FSDP launch requires {required_nodes} nodes, got {topology.nodes}"
            )


def validate_collectives(*, backend: str | None = None, timeout_seconds: int = 120) -> dict:
    """Initialize from torchrun env and prove all ranks participate correctly."""
    topology = topology_from_environment()
    selected = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    if selected == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL validation requires CUDA")
        torch.cuda.set_device(topology.local_rank)
    if not dist.is_initialized():
        dist.init_process_group(selected, timeout=timedelta(seconds=timeout_seconds))
    device = torch.device(f"cuda:{topology.local_rank}") if selected == "nccl" else torch.device("cpu")
    total = torch.tensor(float(topology.rank + 1), device=device)
    dist.all_reduce(total)
    expected = topology.world_size * (topology.world_size + 1) / 2
    if total.item() != expected:
        raise RuntimeError(f"collective result {total.item()} did not equal {expected}")
    hosts: list[str | None] = [None] * topology.world_size
    dist.all_gather_object(hosts, socket.gethostname())
    dist.barrier()
    return {**topology.to_dict(), "backend": selected, "collective_sum": total.item(),
            "hosts": hosts, "observed_nodes": len(set(hosts))}


def validate_3d_parallel_contract(topology: DistributedTopology, *, tensor_parallel: int = 1, pipeline_parallel: int = 1, expert_parallel: int = 1, attention_heads: int = 8, num_experts: int = 1) -> dict:
    from inference.tensor_parallel import ParallelTopologyContract
    contract=ParallelTopologyContract(tensor_parallel, pipeline_parallel, expert_parallel)
    contract.validate(attention_heads=attention_heads, num_experts=num_experts)
    if topology.world_size != contract.world_size:
        raise ValueError(f"3D parallel world size {topology.world_size} does not match contract {contract.world_size}")
    return {"world_size": topology.world_size, "tensor_parallel": tensor_parallel, "pipeline_parallel": pipeline_parallel, "expert_parallel": expert_parallel}
