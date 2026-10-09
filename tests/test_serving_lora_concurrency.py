from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from inference.generator import GenerationStep
from serving.backend import ConfiguredModelBackend
from serving.lora_registry import LoRAAdapterRegistry


class MockGeneratorWithLoRA:
    def __init__(self):
        self.current_adapter = None
        self.states_seen = []
        self.decode_calls = []
        self.tokenizer = MagicMock()
        self.tokenizer.encode.return_value = [1, 2, 3]

    def swap_lora_adapter(self, state):
        self.current_adapter = state
        self.states_seen.append(state)

    def start_batched_stream(self, *_args, **_kwargs):
        return object()

    def release_batched_stream(self, _state):
        return None

    def decode_batched_stream(self, states):
        self.decode_calls.append((self.current_adapter, len(states)))
        return [
            (GenerationStep("x", 1, 3, 1), False)
            for _state in states
        ]


@pytest.mark.asyncio
async def test_tenant_isolated_concurrent_lora_execution():
    generator = MockGeneratorWithLoRA()
    registry = LoRAAdapterRegistry()

    # Register adapters for tenant 1 and tenant 2
    registry.publish("tenant_1", "adapter_1", "v1", {"lora_a": "weights_t1"})
    registry.publish("tenant_2", "adapter_2", "v1", {"lora_b": "weights_t2"})
    registry.activate("tenant_1", "adapter_1")
    registry.activate("tenant_2", "adapter_2")

    # Create dummy backend
    backend = MagicMock(spec=ConfiguredModelBackend)
    backend.generator = generator
    backend.lora_registry = registry
    backend._lora_lock = asyncio.Lock()
    backend._active_lora_tenant_record = None

    # Wire the real scoped context method from ConfiguredModelBackend
    backend._scoped_lora_context = ConfiguredModelBackend._scoped_lora_context.__get__(
        backend, ConfiguredModelBackend
    )

    observed_adapters = {}

    async def simulate_tenant_generate(tenant_id: str):
        async with backend._scoped_lora_context(tenant_id):
            # Record what adapter was active during execution
            adapter_during_exec = generator.current_adapter
            await asyncio.sleep(0.02)  # Simulate generation latency
            # Assert adapter did not change mid-flight
            assert generator.current_adapter == adapter_during_exec
            observed_adapters[tenant_id] = adapter_during_exec

    # Run concurrent generations for tenant_1, tenant_2, and tenant_3 (default/no adapter)
    await asyncio.gather(
        simulate_tenant_generate("tenant_1"),
        simulate_tenant_generate("tenant_2"),
        simulate_tenant_generate("tenant_3"),
    )

    # Verify each tenant observed exactly their own adapter during generation
    assert observed_adapters["tenant_1"] == {"lora_a": "weights_t1"}
    assert observed_adapters["tenant_2"] == {"lora_b": "weights_t2"}
    assert observed_adapters["tenant_3"] is None


@pytest.mark.asyncio
async def test_token_step_stream_batches_only_matching_lora_snapshots():
    generator = MockGeneratorWithLoRA()
    registry = LoRAAdapterRegistry()
    registry.publish("tenant_1", "adapter_1", "v1", {"lora_a": "weights_t1"})
    registry.publish("tenant_2", "adapter_2", "v1", {"lora_b": "weights_t2"})
    registry.activate("tenant_1", "adapter_1")
    registry.activate("tenant_2", "adapter_2")

    backend = ConfiguredModelBackend()
    backend.generator = generator
    backend.lora_registry = registry
    backend._lora_lock = asyncio.Lock()
    backend._active_lora_tenant_record = None

    async def prepare(request):
        return request.prompt, []

    backend._prepare_user_prompt = prepare
    backend._format_request_conversation = lambda _request, _system, prompt: prompt
    backend._validate_prompt = lambda _prompt: [1, 2, 3]

    from serving.schemas import GenerateRequest

    first = await backend.start_stream(GenerateRequest(prompt="one", tenant_id="tenant_1"))
    assert generator.current_adapter == {"lora_a": "weights_t1"}

    second = await backend.start_stream(GenerateRequest(prompt="two", tenant_id="tenant_2"))
    assert generator.current_adapter == {"lora_b": "weights_t2"}
    third = await backend.start_stream(GenerateRequest(prompt="three", tenant_id="tenant_1"))
    await backend.decode_stream_batch([first, second, third])
    assert generator.decode_calls == [
        ({"lora_a": "weights_t1"}, 2),
        ({"lora_b": "weights_t2"}, 1),
    ]
    await backend.release_stream(first)
    await backend.release_stream(second)
    await backend.release_stream(third)
