"""Pipeline-parallel serving runtime with bounded in-flight microbatches."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ServingPipelineConfig:
    stages: int = 1
    max_in_flight: int = 8

    def __post_init__(self):
        if self.stages < 1 or self.max_in_flight < 1:
            raise ValueError("stages and max_in_flight must be positive")


class PipelineServingRuntime:
    def __init__(self, stages: list[Callable[[Any], Any]], *, max_in_flight: int = 8):
        if not stages:
            raise ValueError("at least one stage is required")
        self.stages = tuple(stages)
        self.executor = ThreadPoolExecutor(max_workers=max_in_flight)

    def generate(self, payload):
        value = payload
        for stage in self.stages:
            value = stage(value)
        return value

    def submit(self, payload):
        return self.executor.submit(self.generate, payload)

    def close(self):
        self.executor.shutdown(wait=True)
