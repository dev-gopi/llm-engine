"""Optional DeepSpeed/ZeRO integration without making DeepSpeed a hard dependency."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ZeROConfig:
    stage: int = 2
    offload_optimizer: bool = False
    offload_parameters: bool = False
    overlap_communication: bool = True
    contiguous_gradients: bool = True
    reduce_bucket_size: int | str = "auto"
    stage3_prefetch_bucket_size: int | str = "auto"
    stage3_param_persistence_threshold: int | str = "auto"

    def __post_init__(self):
        if self.stage not in {0, 1, 2, 3}:
            raise ValueError("ZeRO stage must be 0, 1, 2 or 3")
        if self.offload_parameters and self.stage != 3:
            raise ValueError("parameter offload is only supported by ZeRO stage 3")


def available() -> bool:
    try:
        import deepspeed  # type: ignore # noqa: F401

        return True
    except ImportError:
        return False


def build_config(
    cfg: ZeROConfig,
    *,
    micro_batch_size: int,
    gradient_accumulation_steps: int = 1,
    gradient_clipping: float = 1.0,
    mixed_precision: str = "bf16",
) -> dict[str, Any]:
    if micro_batch_size < 1 or gradient_accumulation_steps < 1:
        raise ValueError("batch sizes must be positive")
    if mixed_precision not in {"none", "fp16", "bf16"}:
        raise ValueError("invalid mixed_precision")
    zero: dict[str, Any] = {
        "stage": cfg.stage,
        "overlap_comm": cfg.overlap_communication,
        "contiguous_gradients": cfg.contiguous_gradients,
        "reduce_bucket_size": cfg.reduce_bucket_size,
    }
    if cfg.offload_optimizer:
        zero["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
    if cfg.offload_parameters:
        zero["offload_param"] = {"device": "cpu", "pin_memory": True}
    if cfg.stage == 3:
        zero.update(
            stage3_prefetch_bucket_size=cfg.stage3_prefetch_bucket_size,
            stage3_param_persistence_threshold=cfg.stage3_param_persistence_threshold,
        )
    out: dict[str, Any] = {
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": float(gradient_clipping),
        "zero_optimization": zero,
        "steps_per_print": 0,
        "wall_clock_breakdown": False,
    }
    out["bf16"] = {"enabled": mixed_precision == "bf16"}
    out["fp16"] = {"enabled": mixed_precision == "fp16", "loss_scale": 0}
    return out


class MeshMPU:
    """Minimal DeepSpeed model/data parallel unit backed by ``ParallelMesh``."""

    def __init__(self, mesh):
        self.mesh = mesh

    def get_data_parallel_group(self):
        return self.mesh.group("data")

    def get_model_parallel_group(self):
        return self.mesh.model_group()

    def get_data_parallel_rank(self):
        return self.mesh.local_rank("data")

    def get_data_parallel_world_size(self):
        return self.mesh.degree("data")

    def get_model_parallel_rank(self):
        return self.mesh.model_parallel_rank()

    def get_model_parallel_world_size(self):
        return self.mesh.model_parallel_size()


def validate_pipeline_zero_compatibility(
    *, pipeline_parallel_size: int, zero_stage: int
) -> None:
    """Validate DeepSpeed ZeRO use with the engine's native pipeline runtime.

    DeepSpeed pipeline parallelism is compatible with ZeRO stages 0/1. Stages
    2/3 partition gradients/parameters in ways that require DeepSpeed's own
    pipeline engine rather than this project's explicit activation transport.
    """
    if pipeline_parallel_size < 1:
        raise ValueError("pipeline_parallel_size must be positive")
    if zero_stage not in {0, 1, 2, 3}:
        raise ValueError("ZeRO stage must be 0, 1, 2 or 3")
    if pipeline_parallel_size > 1 and zero_stage > 1:
        raise ValueError(
            "native pipeline parallelism supports DeepSpeed ZeRO stages 0/1 only; "
            "ZeRO-2/3 require DeepSpeed's PipelineEngine"
        )


def initialize(
    model,
    optimizer=None,
    *,
    config: dict[str, Any],
    scheduler=None,
    model_parameters=None,
    mpu=None,
):
    if not available():
        raise RuntimeError(
            "DeepSpeed is not installed; install the optional deepspeed dependency"
        )
    import deepspeed  # type: ignore

    engine, resolved_optimizer, _, resolved_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        model_parameters=model_parameters,
        config=config,
        mpu=mpu,
    )
    return engine, resolved_optimizer, resolved_scheduler


def save_checkpoint(
    engine,
    path: str | Path,
    *,
    tag: str = "latest",
    client_state: dict[str, Any] | None = None,
) -> Path:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    ok = engine.save_checkpoint(str(root), tag=tag, client_state=client_state or {})
    if ok is False:
        raise RuntimeError("DeepSpeed checkpoint save failed")
    return root


def load_checkpoint(
    engine,
    path: str | Path,
    *,
    tag: str | None = None,
    load_optimizer_states: bool = True,
    load_lr_scheduler_states: bool = True,
) -> dict[str, Any]:
    load_path, client_state = engine.load_checkpoint(
        str(path),
        tag=tag,
        load_optimizer_states=load_optimizer_states,
        load_lr_scheduler_states=load_lr_scheduler_states,
    )
    if not load_path:
        raise FileNotFoundError(f"DeepSpeed checkpoint could not be loaded from {path}")
    return dict(client_state or {})


def backward(engine, loss) -> None:
    engine.backward(loss)


def step(engine) -> None:
    engine.step()


__all__ = [
    "ZeROConfig",
    "MeshMPU",
    "available",
    "build_config",
    "validate_pipeline_zero_compatibility",
    "initialize",
    "save_checkpoint",
    "load_checkpoint",
    "backward",
    "step",
]
