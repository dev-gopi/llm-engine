"""Optional inference backend adapters with strict capability detection.

The core engine never silently falls back to another backend. Optional vendor
runtimes are imported lazily and unavailable runtimes raise a typed error.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import Any, Mapping


class BackendUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class BackendCapability:
    name: str
    available: bool
    reason: str | None = None
    version: str | None = None


def detect_backend(name: str) -> BackendCapability:
    name = name.lower().strip()
    modules = {
        "vllm": "vllm",
        "tensorrt_llm": "tensorrt_llm",
        "flashinfer": "flashinfer",
        "flash_attn": "flash_attn",
        "flashattention": "flash_attn",
    }
    if name not in modules:
        raise ValueError(f"unknown inference backend: {name}")
    module = modules[name]
    spec = importlib.util.find_spec(module)
    if spec is None:
        return BackendCapability(name, False, f"optional dependency {module!r} is not installed")
    try:
        mod = __import__(module)
        return BackendCapability(name, True, version=getattr(mod, "__version__", None))
    except Exception as exc:  # pragma: no cover - environment dependent
        return BackendCapability(name, False, f"dependency import failed: {exc}")


class OptionalInferenceBackend:
    """Stable adapter contract for third-party inference engines."""

    name = "base"

    def __init__(self, **options: Any) -> None:
        self.options = dict(options)
        capability = detect_backend(self.name)
        if not capability.available:
            raise BackendUnavailableError(f"{self.name}: {capability.reason}")
        self.capability = capability

    def generate(self, prompts: list[str], **kwargs: Any) -> Any:
        raise NotImplementedError


class VLLMBackend(OptionalInferenceBackend):
    name = "vllm"

    def __init__(self, model: str, **options: Any) -> None:
        self.model = model
        super().__init__(**options)
        from vllm import LLM  # type: ignore
        self._engine = LLM(model=model, **options)

    def generate(self, prompts: list[str], **kwargs: Any) -> Any:
        from vllm import SamplingParams  # type: ignore
        params = SamplingParams(**kwargs)
        return self._engine.generate(prompts, params)


class TensorRTLLMBackend(OptionalInferenceBackend):
    name = "tensorrt_llm"

    def __init__(self, engine: str, **options: Any) -> None:
        self.engine = engine
        super().__init__(**options)
        try:
            import tensorrt_llm  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise BackendUnavailableError(str(exc)) from exc
        self._runtime = tensorrt_llm

    def generate(self, prompts: list[str], **kwargs: Any) -> Any:
        raise NotImplementedError(
            "TensorRT-LLM generation requires an engine-specific tokenizer/runner; "
            "use the adapter's capability contract rather than a silent fallback."
        )


class FlashInferBackend(OptionalInferenceBackend):
    name = "flashinfer"

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        import flashinfer  # type: ignore
        self.runtime = flashinfer


class FlashAttentionBackend(OptionalInferenceBackend):
    name = "flash_attn"

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        import flash_attn  # type: ignore
        self.runtime = flash_attn


def backend_matrix() -> dict[str, BackendCapability]:
    return {name: detect_backend(name) for name in ("vllm", "tensorrt_llm", "flashinfer", "flash_attn")}


__all__ = [
    "BackendCapability", "BackendUnavailableError", "OptionalInferenceBackend",
    "VLLMBackend", "TensorRTLLMBackend", "FlashInferBackend", "FlashAttentionBackend",
    "backend_matrix", "detect_backend",
]
