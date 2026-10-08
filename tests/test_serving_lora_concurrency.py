from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from serving.backend import ConfiguredModelBackend
from serving.lora_registry import LoRAAdapterRegistry


class MockGeneratorWithLoRA:
    def __init__(self):
        self.current_adapter = None
        self.states_seen = []
        self.tokenizer = MagicMock()
        self.tokenizer.encode.return_value = [1, 2, 3]

    def swap_lora_adapter(self, state):
        self.current_adapter = state
        self.states_seen.append(state)


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
