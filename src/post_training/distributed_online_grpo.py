"""Distributed/asynchronous online-GRPO rollout coordination."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import torch.distributed as dist


@dataclass(frozen=True)
class RolloutShard:
    rank: int
    world_size: int
    indices: tuple[int, ...]


def shard_prompts(prompts: Iterable[Any], rank: int, world_size: int) -> list[Any]:
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid rank/world_size")
    return [p for i, p in enumerate(prompts) if i % world_size == rank]


def aggregate_rollouts(
    local_records: list[dict[str, Any]], group=None
) -> list[dict[str, Any]]:
    if (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_world_size(group) == 1
    ):
        return list(local_records)
    gathered = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, local_records, group=group)
    merged = []
    for part in gathered:
        merged.extend(part or [])
    return merged


def async_rollouts(
    records: Iterable[Any],
    worker: Callable[[Any], dict[str, Any]],
    *,
    max_workers: int = 4,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    success, failures = [], []
    with ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="grpo-rollout"
    ) as pool:
        futures = {pool.submit(worker, record): i for i, record in enumerate(records)}
        for future in as_completed(futures):
            i = futures[future]
            try:
                success.append(future.result())
            except Exception as exc:
                failures.append(
                    {"index": i, "error": type(exc).__name__, "message": str(exc)}
                )
    success.sort(key=lambda x: x.get("prompt_index", 0))
    failures.sort(key=lambda x: x["index"])
    return success, failures


def recover_after_rank_failure(
    failed_rank: int, world_size: int, pending: list[Any]
) -> dict[str, Any]:
    if not 0 <= failed_rank < world_size:
        raise ValueError("failed_rank outside world")
    # Deterministic reassignment: next healthy rank receives the failed rank's shard.
    replacement = (failed_rank + 1) % world_size if world_size > 1 else 0
    return {
        "failed_rank": failed_rank,
        "replacement_rank": replacement,
        "pending": list(pending),
        "recoverable": bool(pending),
    }
