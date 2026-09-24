"""Resource discovery and CUDA OOM classification helpers."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

try:
    import torch
except Exception:  # optional runtime dependency in minimal tooling contexts
    torch = None  # type: ignore

from .errors import GPUOutOfMemoryError


@dataclass(frozen=True, slots=True)
class GPUInfo:
    index: int
    name: str
    total_memory_bytes: int
    capability: tuple[int, int] | None


def enumerate_gpus() -> list[GPUInfo]:
    if torch is None or not torch.cuda.is_available():
        return []
    result: list[GPUInfo] = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        result.append(GPUInfo(i, props.name, int(props.total_memory), (props.major, props.minor)))
    return result


def gpu_report() -> list[dict[str, Any]]:
    return [asdict(v) for v in enumerate_gpus()]


def classify_runtime_error(error: BaseException) -> BaseException:
    text = str(error).lower()
    if "out of memory" in text and ("cuda" in text or "gpu" in text):
        return GPUOutOfMemoryError()
    return error
