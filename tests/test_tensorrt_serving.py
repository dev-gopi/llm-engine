from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import inference.backend_adapters as backend_adapters
from serving.backend import TensorRTLLMServingBackend
from serving.schemas import FinishReason, GenerateRequest


class DummyTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(i) for i in token_ids)


class DummyRunner:
    def generate(self, encoded_prompts: list[list[int]], **kwargs) -> list[list[int]]:
        return [list(p) + [33, 34] for p in encoded_prompts]


@pytest.mark.asyncio
async def test_tensorrt_llm_serving_backend(monkeypatch):
    import sys

    dummy_trt = MagicMock()
    monkeypatch.setitem(sys.modules, "tensorrt_llm", dummy_trt)
    monkeypatch.setitem(sys.modules, "tensorrt_llm.runtime", dummy_trt)
    monkeypatch.setattr(
        backend_adapters,
        "detect_backend",
        lambda name: backend_adapters.BackendCapability(name=name, available=True),
    )

    tokenizer = DummyTokenizer()
    runner = DummyRunner()

    backend = TensorRTLLMServingBackend(
        engine="dummy_engine_dir",
        tokenizer=tokenizer,
        runner=runner,
    )
    assert backend.ready is True

    req = GenerateRequest(prompt="hi", max_tokens=10)
    gen = await backend.generate(req)
    assert gen.text == 'hi!"'
    assert gen.prompt_tokens == 2
    assert gen.completion_tokens == 4
    assert gen.finish_reason == FinishReason.STOP

    events = []
    async for ev in backend.stream(req):
        events.append(ev)
    assert len(events) == 1
    assert events[0].token == 'hi!"'


def test_tensorrt_llm_reload_candidate(monkeypatch, tmp_path):
    import sys

    dummy_trt = MagicMock()
    dummy_runner_cls = MagicMock()
    dummy_trt.runtime.ModelRunner = dummy_runner_cls
    monkeypatch.setitem(sys.modules, "tensorrt_llm", dummy_trt)
    monkeypatch.setitem(sys.modules, "tensorrt_llm.runtime", dummy_trt.runtime)
    monkeypatch.setattr(
        backend_adapters,
        "detect_backend",
        lambda name: backend_adapters.BackendCapability(name=name, available=True),
    )

    from serving.backend import _reload_candidate

    monkeypatch.setenv("GOPI_BACKEND", "tensorrt_llm")
    monkeypatch.setenv("GOPI_TRT_ENGINE", str(tmp_path))

    backend, version = _reload_candidate()
    assert version.startswith("tensorrt_llm:")
    assert isinstance(backend, TensorRTLLMServingBackend)
