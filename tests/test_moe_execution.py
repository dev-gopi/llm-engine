from __future__ import annotations

import torch

from runtime.moe_execution import (
    build_executable_moe_model,
    execute_moe_forward,
    execute_sparse_profile,
)


def test_moe_execution_100b_profile():
    result = execute_sparse_profile(
        "configs/scaling/model.moe-100b.yaml",
        sequence_length=8,
        batch_size=1,
        device="meta",
    )
    assert result["executed"] is True
    assert result["ffn_type"] == "moe"
    assert result["total_parameters"] > 90_000_000_000
    assert result["active_parameters"] < 20_000_000_000
    assert result["output_shape"] == [1, 8, 50000]


def test_hybrid_moe_7b_profile_execution():
    result = execute_sparse_profile(
        "configs/scaling/model.hybrid-moe-7b.yaml",
        sequence_length=8,
        batch_size=1,
        device="meta",
    )
    assert result["executed"] is True
    assert result["ffn_type"] == "moe"
    assert 6_000_000_000 < result["total_parameters"] < 10_000_000_000
    assert result["output_shape"] == [1, 8, 50000]


def test_small_moe_forward_cpu():
    config = {
        "vocab_size": 32,
        "hidden_size": 16,
        "layers": 2,
        "heads": 2,
        "kv_heads": 2,
        "max_position": 32,
        "ffn_type": "moe",
        "ffn_hidden_size": 32,
        "num_experts": 4,
        "experts_per_token": 2,
        "planning_only": True,
    }
    # With allow_planning=True, execution succeeds even when planning_only=True
    model = build_executable_moe_model(config, device="cpu", dtype=torch.float32)
    inputs = torch.randint(0, 32, (1, 4))
    logits = execute_moe_forward(model, inputs)
    assert logits.shape == (1, 4, 32)
