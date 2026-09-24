"""Resource-aware GPU selection and guarded execution helpers."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable

from .errors import GPUOutOfMemoryError, ModelNotAvailableError
from .resources import gpu_report


@dataclass(frozen=True, slots=True)
class ResourceRequest:
    minimum_vram_bytes: int = 0
    preferred_device: int | None = None
    allow_cpu: bool = False


class GPUScheduler:
    def __init__(self, *, max_concurrency_per_gpu: int = 1) -> None:
        self.max_concurrency_per_gpu = max(1, max_concurrency_per_gpu)
        self._locks: dict[int, threading.BoundedSemaphore] = {}
        self._guard = threading.Lock()

    def select(self, request: ResourceRequest) -> str:
        gpus = gpu_report()
        if request.preferred_device is not None:
            gpus = [g for g in gpus if int(g.get("index", -1)) == request.preferred_device]
        compatible = [g for g in gpus if int(g.get("total_memory_bytes") or 0) >= request.minimum_vram_bytes]
        if compatible:
            compatible.sort(key=lambda g: int(g.get("total_memory_bytes") or 0), reverse=True)
            return f"cuda:{int(compatible[0]['index'])}"
        if request.allow_cpu:
            return "cpu"
        raise ModelNotAvailableError("no compatible GPU satisfies the requested VRAM budget")

    def run(self, device: str, fn: Callable[[], Any]) -> Any:
        if not device.startswith("cuda:"):
            return fn()
        index = int(device.split(":", 1)[1])
        with self._guard:
            sem = self._locks.setdefault(index, threading.BoundedSemaphore(self.max_concurrency_per_gpu))
        with sem:
            try:
                return fn()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and "cuda" in str(exc).lower():
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    raise GPUOutOfMemoryError("CUDA out of memory during model execution") from exc
                raise
