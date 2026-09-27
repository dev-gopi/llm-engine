"""Pipeline-parallel scheduling and point-to-point distributed execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class PipelineSchedule:
    stages: int
    microbatches: int

    def __post_init__(self):
        if self.stages < 1 or self.microbatches < 1:
            raise ValueError("stages and microbatches must be positive")

    def one_forward_one_backward(self):
        events = []
        for t in range(self.microbatches + self.stages - 1):
            for stage in range(self.stages):
                mb = t - stage
                if 0 <= mb < self.microbatches:
                    events.append((mb, stage, "forward"))
            for stage in range(self.stages - 1, -1, -1):
                mb = t - (self.stages - 1 - stage)
                if 0 <= mb < self.microbatches and t >= self.stages - 1:
                    events.append((mb, stage, "backward"))
        return events


def run_pipeline(
    stages, microbatches, inputs, *, loss_fn: Callable[[Any], Any] | None = None
):
    if len(stages) < 1:
        raise ValueError("at least one stage required")
    outputs = []
    for item in inputs:
        x = item
        for stage in stages:
            x = stage(x)
        outputs.append(x)
    if loss_fn is not None:
        return outputs, [loss_fn(x) for x in outputs]
    return outputs


class DistributedPipelineStage:
    """One synchronous pipeline stage with explicit activation/gradient transport.

    Shape and dtype are static by design.  This keeps the transport deterministic
    and avoids pickling tensors. ``train_step`` performs true cross-rank autograd by
    sending activation gradients back to the previous stage.
    """

    def __init__(
        self,
        module,
        *,
        stage: int,
        stages: int,
        activation_shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        group=None,
        device=None,
    ) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("pipeline parallelism requires torch.distributed")
        if not 0 <= stage < stages:
            raise ValueError("invalid pipeline stage")
        if any(int(x) < 1 for x in activation_shape):
            raise ValueError("activation_shape must be positive")
        self.module = module
        self.stage = stage
        self.stages = stages
        self.group = dist.group.WORLD if group is None else group
        self.members = list(range(dist.get_world_size(self.group)))
        if stages != len(self.members):
            raise ValueError("pipeline stages must equal pipeline process-group size")
        self.shape = tuple(int(x) for x in activation_shape)
        self.dtype = dtype
        self.device = torch.device(device or next(module.parameters()).device)

    @property
    def first(self):
        return self.stage == 0

    @property
    def last(self):
        return self.stage == self.stages - 1

    def forward(self, value: torch.Tensor | None = None) -> torch.Tensor:
        if self.first:
            if value is None:
                raise ValueError("first pipeline stage requires input")
            x = value.to(self.device)
        else:
            x = torch.empty(self.shape, device=self.device, dtype=self.dtype)
            dist.recv(x, src=self.members[self.stage - 1], group=self.group)
            x.requires_grad_(True)
        y = self.module(x)
        if not self.last:
            if tuple(y.shape) != self.shape:
                raise ValueError("pipeline activation shape changed between stages")
            dist.send(
                y.detach().contiguous(),
                dst=self.members[self.stage + 1],
                group=self.group,
            )
        self._input = x
        self._output = y
        return y

    def backward(self, loss: torch.Tensor | None = None) -> torch.Tensor | None:
        if self.last:
            if loss is None:
                raise ValueError("last pipeline stage requires a loss")
            loss.backward()
        else:
            grad = torch.empty(self.shape, device=self.device, dtype=self.dtype)
            dist.recv(grad, src=self.members[self.stage + 1], group=self.group)
            self._output.backward(grad)
        if not self.first:
            input_grad = self._input.grad
            if input_grad is None:
                raise RuntimeError("pipeline input gradient was not produced")
            dist.send(
                input_grad.detach().contiguous(),
                dst=self.members[self.stage - 1],
                group=self.group,
            )
            return input_grad
        return None


__all__ = ["PipelineSchedule", "run_pipeline", "DistributedPipelineStage"]
