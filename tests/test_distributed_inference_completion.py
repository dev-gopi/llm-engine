import pytest
import torch

from inference.backend_routing import (
    BackendEndpoint,
    CircuitBreaker,
    CircuitState,
    FailoverRouter,
    TenantQoSQueue,
)
from inference.quantized_formats import convert_native
from training.context_parallel import context_parallel_causal_mask
from training.deepspeed import ZeROConfig, build_config
from training.expert_parallel import (
    expert_indices,
    return_tokens_to_sources,
    route_tokens_to_expert_owners,
)
from training.parallelism import ParallelDegrees


def test_zero3_config_has_offload_and_precision():
    cfg = build_config(
        ZeROConfig(stage=3, offload_optimizer=True, offload_parameters=True),
        micro_batch_size=2,
        gradient_accumulation_steps=4,
        mixed_precision="bf16",
    )
    assert cfg["zero_optimization"]["stage"] == 3
    assert cfg["zero_optimization"]["offload_param"]["device"] == "cpu"
    assert cfg["bf16"]["enabled"] is True
    assert cfg["fp16"]["enabled"] is False


def test_parallel_degrees_world_size_and_validation():
    degrees = ParallelDegrees(
        data=2, tensor=2, pipeline=2, expert=2, context=1, sequence=1
    )
    assert degrees.world_size == 16
    with pytest.raises(ValueError):
        ParallelDegrees(tensor=0)


def test_context_parallel_mask():
    mask = context_parallel_causal_mask(2, 4, rank=1, world_size=2)
    assert mask.tolist() == [[True, True, True, False], [True, True, True, True]]


def test_expert_exchange_single_rank_roundtrip():
    x = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    ids = torch.tensor([0, 1, 0, 1])
    # one rank requires num_experts divisible by one and is a no-op transport
    routed, rids, meta = route_tokens_to_expert_owners(x, ids, num_experts=2)
    assert sorted(rids.tolist()) == [0, 0, 1, 1]
    restored = return_tokens_to_sources(routed, meta)
    assert torch.equal(restored, x)
    assert expert_indices(4, 1, 2) == (1, 3)


def test_circuit_breaker_and_failover():
    bad = lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom"))
    good = lambda x: x + 1
    first = BackendEndpoint(
        "bad",
        bad,
        priority=1,
        breaker=CircuitBreaker(failure_threshold=1, recovery_seconds=100),
    )
    second = BackendEndpoint("good", good, priority=2)
    router = FailoverRouter([first, second])
    assert router.generate(3) == 4
    assert first.breaker.state is CircuitState.OPEN
    assert router.generate(4) == 5
    assert first.requests == 1


def test_qos_queue_enforces_tenant_limit():
    q = TenantQoSQueue(default_limit=1)
    q.submit("a", 1)
    q.submit("a", 2)
    q.submit("b", 3)
    one = q.acquire()
    two = q.acquire()
    assert one == ("a", 1)
    assert two == ("b", 3)
    q.release("a")
    assert q.acquire() == ("a", 2)


def test_native_converter_hook_does_not_require_vendor_runtime(tmp_path):
    out = tmp_path / "model.gptq"

    def converter(*, model, output, format, **kwargs):
        output.write_text(format)
        return output

    result = convert_native(fmt="gptq", model=object(), output=out, converter=converter)
    assert result == out and out.read_text() == "gptq"
