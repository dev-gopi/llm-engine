"""Production runtime qualification probes.

These probes never claim a backend is production-qualified merely because a
Python package imports. They record the concrete host/runtime prerequisites so
release automation can require evidence produced on the target GPU/cluster.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class QualificationProbe:
    name: str
    passed: bool
    observed: str
    required: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _module_probe(name: str, module: str) -> QualificationProbe:
    found = importlib.util.find_spec(module) is not None
    return QualificationProbe(
        name=name,
        passed=found,
        observed="installed" if found else "missing",
        required=f"Python module {module}",
    )


def _executable_probe(name: str, executable: str) -> QualificationProbe:
    path = shutil.which(executable)
    return QualificationProbe(
        name=name,
        passed=path is not None,
        observed=path or "missing",
        required=f"executable {executable}",
    )


def runtime_probes() -> list[QualificationProbe]:
    cuda = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda else 0
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    nnodes = int(os.getenv("NNODES", os.getenv("SLURM_NNODES", "1")))
    probes = [
        QualificationProbe("cuda", cuda, f"available={cuda}", "CUDA available"),
        QualificationProbe(
            "gpu_count", gpu_count > 0, str(gpu_count), ">=1 visible CUDA GPU"
        ),
        QualificationProbe(
            "distributed_world",
            world_size > 1,
            str(world_size),
            ">1 process for distributed qualification",
        ),
        QualificationProbe(
            "multi_node",
            nnodes > 1,
            str(nnodes),
            ">1 node for multi-node qualification",
        ),
        _module_probe("deepspeed", "deepspeed"),
        _module_probe("tensorrt_llm", "tensorrt_llm"),
        _module_probe("flash_attention", "flash_attn"),
        _module_probe("flashinfer", "flashinfer"),
        _module_probe("torchao", "torchao"),
        _module_probe("gptq", "auto_gptq"),
        _module_probe("awq", "awq"),
        _module_probe("vllm", "vllm"),
        _executable_probe("llama_server", "llama-server"),
        _executable_probe("llama_quantize", "llama-quantize"),
    ]
    if cuda:
        capability = torch.cuda.get_device_capability(0)
        probes.append(
            QualificationProbe(
                "compute_capability",
                capability >= (8, 0),
                f"{capability[0]}.{capability[1]}",
                ">=8.0 for the default modern CUDA qualification matrix",
            )
        )
    return probes


def qualify_profile(
    profile: Mapping[str, Any], probes: list[QualificationProbe] | None = None
) -> dict[str, Any]:
    """Evaluate a production profile against observations from this runtime."""
    observations = {probe.name: probe for probe in (probes or runtime_probes())}
    requirements = profile.get("qualification") or {}
    if not isinstance(requirements, Mapping):
        raise ValueError("qualification must be a mapping")

    failures: list[str] = []
    if bool(requirements.get("requires_cuda")) and not observations["cuda"].passed:
        failures.append("CUDA is required")
    if (
        bool(requirements.get("requires_multi_node"))
        and not observations["multi_node"].passed
    ):
        failures.append("multi-node execution is required")

    required_nodes = int(requirements.get("required_nodes", 1))
    observed_nodes = (
        int(observations["multi_node"].observed) if "multi_node" in observations else 1
    )
    if required_nodes > observed_nodes:
        failures.append(
            f"requires at least {required_nodes} nodes; observed {observed_nodes}"
        )

    required_memory = float(requirements.get("required_gpu_memory_gb", 0) or 0)
    observed_memory = 0.0
    if torch.cuda.is_available() and torch.cuda.device_count():
        observed_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if required_memory > observed_memory:
        failures.append(
            f"requires at least {required_memory:g} GiB GPU memory; observed {observed_memory:.2f} GiB"
        )

    return {
        "profile": profile.get("profile", "unknown"),
        "qualified": not failures,
        "failures": failures,
        "probes": [probe.to_dict() for probe in observations.values()],
    }


__all__ = ["QualificationProbe", "runtime_probes", "qualify_profile"]
