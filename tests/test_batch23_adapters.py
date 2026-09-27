from __future__ import annotations

import pytest

from inference.backend_adapters import backend_matrix, detect_backend
from inference.quantized_formats import NativeFormatUnavailable, capabilities, require_native
from inference.speculative import DraftModelRegistry, SpeculativeConfig, speculative_accept
from serving.lora_registry import LoRAAdapterRegistry
from omni_platform.voice_cloning import ProviderVoiceCloner, VoiceClonePolicy


def test_backend_detection_is_non_destructive():
    matrix = backend_matrix()
    assert set(matrix) == {"vllm", "tensorrt_llm", "flashinfer", "flash_attn"}
    assert all(item.name for item in matrix.values())


def test_native_format_capability_does_not_lie():
    caps = capabilities()
    assert set(caps) == {"gptq", "awq", "gguf"}
    for fmt, cap in caps.items():
        if not cap.available:
            with pytest.raises(NativeFormatUnavailable):
                require_native(fmt)


def test_speculative_registry_and_prefix_acceptance():
    config = SpeculativeConfig(enabled=True, draft_model_id="draft-v1")
    assert config.max_draft_tokens == 5
    registry = DraftModelRegistry()
    registry.register("draft-v1", object())
    assert registry.ids() == ("draft-v1",)
    accepted, index = speculative_accept([1, 2, 3], lambda token, i: token != 3)
    assert accepted == [1, 2]
    assert index == 2


def test_lora_registry_is_tenant_isolated_and_atomic():
    registry = LoRAAdapterRegistry(max_adapters_per_tenant=1)
    registry.publish("t1", "a", "v1", {"weights": 1})
    registry.publish("t2", "a", "v1", {"weights": 2})
    registry.activate("t1", "a")
    assert registry.active("t1").state["weights"] == 1
    assert registry.active("t2") is None
    with pytest.raises(RuntimeError):
        registry.remove("t1", "a")


def test_voice_clone_policy_requires_consent():
    class Provider:
        id = "test"
        capabilities = frozenset({"voice_cloning"})
        def is_available(self): return True
        def clone_voice(self, request, context): return "ok"

    wrapper = ProviderVoiceCloner(Provider())
    with pytest.raises(PermissionError):
        wrapper.clone_voice({"reference_duration_seconds": 2}, None)
    with pytest.raises(ValueError):
        wrapper.clone_voice({"reference_duration_seconds": 31, "consent_token": "x"}, None)
