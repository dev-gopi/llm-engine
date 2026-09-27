"""Optional DeepSpeed/ZeRO integration without making DeepSpeed a hard dependency."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class ZeROConfig:
    stage: int = 2
    offload_optimizer: bool = False
    offload_parameters: bool = False
    overlap_communication: bool = True
    contiguous_gradients: bool = True
    def __post_init__(self):
        if self.stage not in {1,2,3}: raise ValueError("ZeRO stage must be 1, 2 or 3")

def available() -> bool:
    try:
        import deepspeed  # type: ignore
        return True
    except ImportError:
        return False

def build_config(cfg: ZeROConfig, *, micro_batch_size: int, gradient_accumulation_steps: int = 1) -> dict[str, Any]:
    if micro_batch_size < 1 or gradient_accumulation_steps < 1: raise ValueError("batch sizes must be positive")
    return {"train_micro_batch_size_per_gpu": micro_batch_size, "gradient_accumulation_steps": gradient_accumulation_steps,
            "zero_optimization":{"stage":cfg.stage,"offload_optimizer":{"device":"cpu"} if cfg.offload_optimizer else None,
            "offload_param":{"device":"cpu"} if cfg.offload_parameters else None,
            "overlap_comm":cfg.overlap_communication,"contiguous_gradients":cfg.contiguous_gradients},
            "bf16":{"enabled":"auto"},"gradient_clipping":1.0}

def initialize(model, optimizer, *, config: dict[str,Any]):
    if not available(): raise RuntimeError("DeepSpeed is not installed; install the optional deepspeed dependency")
    import deepspeed  # type: ignore
    engine, optimizer, _, scheduler = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
    return engine, optimizer, scheduler
