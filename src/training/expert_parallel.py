"""Expert-parallel routing, exchange, and checkpoint utilities."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch
import torch.distributed as dist


def expert_indices(num_experts: int, rank: int, world_size: int) -> tuple[int, ...]:
    if (
        num_experts < 1
        or world_size < 1
        or num_experts % world_size
        or not 0 <= rank < world_size
    ):
        raise ValueError("incompatible expert topology")
    return tuple(range(rank, num_experts, world_size))


def expert_owner(expert: int, world_size: int) -> int:
    if expert < 0 or world_size < 1:
        raise ValueError("invalid expert topology")
    return expert % world_size


def save_expert_shard(
    path, state_dict, *, num_experts: int, rank: int, world_size: int
):
    idx = expert_indices(num_experts, rank, world_size)
    prefixes = tuple(f"experts.{i}." for i in idx)
    shard = {k: v for k, v in state_dict.items() if k.startswith(prefixes)}
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "gopi-expert-v1",
            "rank": rank,
            "world_size": world_size,
            "indices": idx,
            "state_dict": shard,
        },
        dst,
    )
    return dst


def load_expert_shard(path, module, *, strict=False):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "gopi-expert-v1":
        raise ValueError("unsupported expert shard format")
    result = module.load_state_dict(payload["state_dict"], strict=False)
    if strict and (result.missing_keys or result.unexpected_keys):
        raise RuntimeError(str(result))
    return payload


def all_to_all_variable(
    tensor: torch.Tensor, send_counts: Iterable[int], *, group=None
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Variable-size all-to-all along dim 0, including count exchange."""
    if not dist.is_available() or not dist.is_initialized():
        counts = tuple(int(x) for x in send_counts)
        if len(counts) != 1:
            raise RuntimeError("multi-rank expert exchange requires torch.distributed")
        return tensor, counts
    group = dist.group.WORLD if group is None else group
    world = dist.get_world_size(group)
    send = tuple(int(x) for x in send_counts)
    if len(send) != world or any(x < 0 for x in send):
        raise ValueError("send_counts must contain one non-negative count per rank")
    if sum(send) != tensor.shape[0]:
        raise ValueError("send_counts do not match tensor rows")
    device = tensor.device
    send_t = torch.tensor(send, dtype=torch.int64, device=device)
    recv_t = torch.empty_like(send_t)
    dist.all_to_all_single(recv_t, send_t, group=group)
    recv = tuple(int(x) for x in recv_t.cpu().tolist())
    output = torch.empty(
        (sum(recv), *tensor.shape[1:]), dtype=tensor.dtype, device=device
    )
    dist.all_to_all_single(
        output,
        tensor.contiguous(),
        output_split_sizes=list(recv),
        input_split_sizes=list(send),
        group=group,
    )
    return output, recv


def route_tokens_to_expert_owners(
    tokens: torch.Tensor, expert_ids: torch.Tensor, *, num_experts: int, group=None
):
    """Route token rows to ranks owning their selected experts.

    Returns ``(received_tokens, received_expert_ids, metadata)``. Metadata can be
    passed to :func:`return_tokens_to_sources` after local expert computation.
    """
    if (
        tokens.ndim < 2
        or expert_ids.ndim != 1
        or tokens.shape[0] != expert_ids.shape[0]
    ):
        raise ValueError("tokens/expert_ids shape mismatch")
    if num_experts < 1:
        raise ValueError("num_experts must be positive")
    world = (
        dist.get_world_size(group)
        if dist.is_available() and dist.is_initialized()
        else 1
    )
    if num_experts % world:
        raise ValueError("num_experts must divide expert-parallel world size")
    if expert_ids.numel() and (
        int(expert_ids.min()) < 0 or int(expert_ids.max()) >= num_experts
    ):
        raise ValueError("expert id outside range")
    owners = (expert_ids % world).to(torch.long)
    order = torch.argsort(owners, stable=True)
    sorted_tokens = tokens.index_select(0, order)
    sorted_experts = expert_ids.index_select(0, order)
    counts = tuple(int((owners == rank).sum()) for rank in range(world))
    recv_tokens, recv_counts = all_to_all_variable(sorted_tokens, counts, group=group)
    recv_experts, _ = all_to_all_variable(
        sorted_experts.reshape(-1, 1), counts, group=group
    )
    return (
        recv_tokens,
        recv_experts.reshape(-1),
        {"order": order, "send_counts": counts, "recv_counts": recv_counts},
    )


def return_tokens_to_sources(
    processed: torch.Tensor, metadata: dict, *, group=None
) -> torch.Tensor:
    """Reverse :func:`route_tokens_to_expert_owners` and restore original row order."""
    recv_counts = tuple(metadata["recv_counts"])
    send_counts = tuple(metadata["send_counts"])
    returned, _ = all_to_all_variable(processed, recv_counts, group=group)
    if returned.shape[0] != sum(send_counts):
        raise RuntimeError("expert return exchange size mismatch")
    order = metadata["order"].to(returned.device)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=order.device)
    return returned.index_select(0, inverse)


__all__ = [
    "all_to_all_variable",
    "expert_indices",
    "expert_owner",
    "load_expert_shard",
    "return_tokens_to_sources",
    "route_tokens_to_expert_owners",
    "save_expert_shard",
]
