"""Composable distributed process meshes for DP/TP/PP/EP/CP/SP.

The module intentionally depends only on torch.distributed so it can be reused by
training and inference.  Groups are created in a deterministic global order on
all ranks, which is required by ``dist.new_group`` and avoids topology deadlocks.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Iterable

import torch.distributed as dist


@dataclass(frozen=True)
class ParallelDegrees:
    data: int = 1
    tensor: int = 1
    pipeline: int = 1
    expert: int = 1
    context: int = 1
    sequence: int = 1

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} parallel degree must be a positive integer")

    @property
    def world_size(self) -> int:
        return reduce(mul, self.__dict__.values(), 1)

    def as_tuple(self) -> tuple[int, ...]:
        return (self.data, self.pipeline, self.tensor, self.expert, self.context, self.sequence)


_AXIS = ("data", "pipeline", "tensor", "expert", "context", "sequence")


class ParallelMesh:
    """Deterministic N-D rank mesh and process-group registry."""

    def __init__(self, degrees: ParallelDegrees, *, world_group=None) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("parallel mesh requires an initialized torch.distributed process group")
        self.degrees = degrees
        self.world_group = dist.group.WORLD if world_group is None else world_group
        self.world_size = dist.get_world_size(self.world_group)
        self.rank = dist.get_rank(self.world_group)
        if degrees.world_size != self.world_size:
            raise ValueError(
                f"parallel degrees multiply to {degrees.world_size}, but process-group world size is {self.world_size}"
            )
        self._groups: dict[str, object] = {}
        self._members: dict[str, tuple[int, ...]] = {}
        # Every rank creates every group in exactly the same order.
        for axis in _AXIS:
            for members in self.groups_for_axis(axis):
                group = dist.new_group(list(members))
                if self.rank in members:
                    self._groups[axis] = group
                    self._members[axis] = members

    def coordinate(self, rank: int | None = None) -> dict[str, int]:
        rank = self.rank if rank is None else int(rank)
        if not 0 <= rank < self.world_size:
            raise ValueError("rank is outside the mesh")
        sizes = self.degrees.as_tuple()
        coords: dict[str, int] = {}
        remainder = rank
        for axis, size in reversed(list(zip(_AXIS, sizes))):
            coords[axis] = remainder % size
            remainder //= size
        return {axis: coords[axis] for axis in _AXIS}

    def rank_at(self, **coordinates: int) -> int:
        sizes = self.degrees.as_tuple()
        rank = 0
        for axis, size in zip(_AXIS, sizes):
            value = int(coordinates.get(axis, 0))
            if not 0 <= value < size:
                raise ValueError(f"{axis} coordinate is outside the mesh")
            rank = rank * size + value
        return rank

    def groups_for_axis(self, axis: str) -> tuple[tuple[int, ...], ...]:
        if axis not in _AXIS:
            raise ValueError(f"unknown parallel axis {axis!r}")
        target = _AXIS.index(axis)
        buckets: dict[tuple[int, ...], list[int]] = {}
        for rank in range(self.world_size):
            coord = self.coordinate(rank)
            key = tuple(coord[name] for index, name in enumerate(_AXIS) if index != target)
            buckets.setdefault(key, []).append(rank)
        return tuple(tuple(value) for _, value in sorted(buckets.items()))

    def group(self, axis: str):
        return self._groups[axis]

    def members(self, axis: str) -> tuple[int, ...]:
        return self._members[axis]

    def local_rank(self, axis: str) -> int:
        return self.members(axis).index(self.rank)

    def degree(self, axis: str) -> int:
        return int(getattr(self.degrees, axis))

    def barrier(self, axes: Iterable[str] | None = None) -> None:
        for axis in (axes or _AXIS):
            if self.degree(axis) > 1:
                dist.barrier(group=self.group(axis))


__all__ = ["ParallelDegrees", "ParallelMesh"]
