import asyncio

import pytest
import torch
from torch import nn

from diffusion.pipeline import DiffusionPipeline
from diffusion.scheduler import DiffusionScheduler
from diffusion.unet import SmallUNet
from embeddings.service import EmbeddingService
from inference.agent_runtime import AgentState, AgentStep, BoundedAgentRuntime
from inference.tensor_parallel import ParallelTopologyContract
from inference.web_search import search_brave
from model.feed_forward import SparseMoE
from multimodal.projector import VisionProjector
from optim.ema import EMA
from rag.reranker import HybridReranker, LexicalCrossEncoderBaseline
from serving.batching import DynamicBatcher
from serving.rate_limit import InMemoryRateLimiter


def test_sparse_moe_capacity_metrics_and_router_z_loss() -> None:
    torch.manual_seed(2)
    moe = SparseMoE(
        dim=8,
        hidden_dim=16,
        num_experts=2,
        experts_per_token=2,
        capacity_factor=0.25,
        min_capacity=1,
    )
    output = moe(torch.randn(2, 4, 8))
    assert output.shape == (2, 4, 8)
    assert moe.last_router_z_loss is not None and torch.isfinite(moe.last_router_z_loss)
    assert moe.last_expert_capacity == 2
    assert 0 < moe.last_dropped_route_fraction < 1
    metrics = moe.routing_metrics()
    assert metrics["expert_capacity"] == 2


def test_rate_limiter_exposes_retry_metadata() -> None:
    async def scenario():
        limiter = InMemoryRateLimiter(1)
        first = await limiter.check("client")
        second = await limiter.check("client")
        return first, second

    first, second = asyncio.run(scenario())
    assert first.allowed and first.remaining == 0
    assert not second.allowed and second.retry_after_seconds > 0


def test_dynamic_batcher_drops_cancelled_queued_work() -> None:
    class Backend:
        calls = []

        async def batch_generate(self, requests):
            self.calls.append(list(requests))
            return list(requests)

    async def scenario():
        backend = Backend()
        batcher = DynamicBatcher(backend, wait_milliseconds=10)
        await batcher.startup()
        cancelled = asyncio.create_task(batcher.generate("cancelled"))
        active = asyncio.create_task(batcher.generate("active"))
        await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        result = await active
        await batcher.shutdown()
        return backend.calls, result

    calls, result = asyncio.run(scenario())
    assert result == "active"
    assert calls == [["active"]]


def test_hybrid_reranker_fails_open_to_lexical_baseline() -> None:
    class Broken:
        def rerank(self, query, documents, *, top_k=5):
            raise RuntimeError("offline")

    class Doc:
        def __init__(self, text):
            self.text = text

    docs = [Doc("quantum field theory"), Doc("cats are mammals")]
    ranked = HybridReranker(Broken()).rerank("cats", docs, top_k=1)
    assert ranked[0].document is docs[1]


def test_diffusion_min_snr_loss_and_negative_condition_validation() -> None:
    model = SmallUNet(image_channels=3, base_channels=8, condition_size=16)
    pipeline = DiffusionPipeline(model, DiffusionScheduler(timesteps=4))
    loss = pipeline.training_loss(
        torch.randn(2, 3, 8, 8),
        torch.randn(2, 16),
        min_snr_gamma=5.0,
    )
    assert torch.isfinite(loss)
    with pytest.raises(ValueError, match="negative_text_condition"):
        pipeline.sample(
            1,
            8,
            device="cpu",
            text_condition=torch.randn(1, 16),
            negative_text_condition=torch.randn(2, 16),
            inference_steps=2,
        )


def test_parallel_contract_validates_kv_head_sharding(monkeypatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    ParallelTopologyContract(tensor_parallel=2).validate(
        attention_heads=8, kv_heads=2, num_experts=4
    )
    with pytest.raises(ValueError, match="KV"):
        ParallelTopologyContract(tensor_parallel=2).validate(
            attention_heads=8, kv_heads=1, num_experts=4
        )


def test_ema_rejects_corrupt_state() -> None:
    ema = EMA(nn.Linear(2, 2))
    state = ema.state_dict()
    state["decay"] = 1.5
    with pytest.raises(ValueError, match="decay"):
        ema.load_state_dict(state)


def test_embedding_service_can_unit_normalize_vectors() -> None:
    result = EmbeddingService().encode(["alpha beta", "gamma delta"], normalize=True)
    vectors = torch.tensor(result.embeddings)
    torch.testing.assert_close(vectors.norm(dim=-1), torch.ones(2), atol=1e-5, rtol=1e-5)


def test_web_search_rejects_non_http_endpoint_before_network_access() -> None:
    with pytest.raises(ValueError, match="HTTP"):
        search_brave("query", "key", endpoint="file:///tmp/search")


def test_agent_supports_stepwise_approval_and_copies_plan() -> None:
    original = AgentStep("echo", {"value": 1})
    runtime = BoundedAgentRuntime({"echo": lambda value: value})
    runtime.plan([original])
    original.arguments["value"] = 99
    runtime.approve_step(0)
    assert runtime.state == AgentState.EXECUTING
    assert runtime.run()[0].result == 1


def test_vision_projector_validates_dropout() -> None:
    with pytest.raises(ValueError, match="dropout"):
        VisionProjector(8, 8, dropout=1.0)
