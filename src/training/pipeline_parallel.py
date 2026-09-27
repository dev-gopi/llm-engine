"""Pipeline-parallel stage scheduling helpers."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any

@dataclass(frozen=True)
class PipelineSchedule:
    stages: int
    microbatches: int
    def __post_init__(self):
        if self.stages<1 or self.microbatches<1: raise ValueError("stages and microbatches must be positive")
    def one_forward_one_backward(self):
        # Deterministic 1F1B schedule represented as (microbatch, stage, phase).
        events=[]
        for t in range(self.microbatches+self.stages-1):
            for stage in range(self.stages):
                mb=t-stage
                if 0<=mb<self.microbatches: events.append((mb,stage,"forward"))
            for stage in range(self.stages-1,-1,-1):
                mb=t-(self.stages-1-stage)
                if 0<=mb<self.microbatches and t>=self.stages-1: events.append((mb,stage,"backward"))
        return events

def run_pipeline(stages, microbatches, inputs, *, loss_fn:Callable[[Any],Any]|None=None):
    if len(stages)<1: raise ValueError("at least one stage required")
    outputs=[]
    for item in inputs:
        x=item
        for stage in stages: x=stage(x)
        outputs.append(x)
    if loss_fn is not None: return outputs, [loss_fn(x) for x in outputs]
    return outputs
