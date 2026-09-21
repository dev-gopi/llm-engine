import pytest

from runtime.resource_planner import deployment_matrix, estimate_inference_memory


def _config():
    return {
        "vocab_size": 128,
        "hidden_size": 64,
        "layers": 8,
        "heads": 8,
        "kv_heads": 2,
        "max_position": 4096,
        "position_type": "rotary",
        "ffn_hidden_size": 128,
        "attention_layer_pattern": ["linear", "linear", "linear", "dense"],
    }


def test_q1_and_int4_kv_reduce_analytical_memory():
    bf16 = estimate_inference_memory(_config(), context_length=2048, weight_precision="bf16", kv_precision="bf16")
    compact = estimate_inference_memory(_config(), context_length=2048, weight_precision="q1_0", kv_precision="int4")
    assert compact.weight_bytes < bf16.weight_bytes
    assert compact.kv_cache_bytes < bf16.kv_cache_bytes
    assert compact.estimated_total_bytes < bf16.estimated_total_bytes
    assert compact.linear_state_bytes > 0


def test_deployment_matrix_is_sorted_and_budget_aware():
    rows = deployment_matrix(_config(), context_length=1024, memory_budget_bytes=50_000_000)
    totals = [row["estimated_total_bytes"] for row in rows]
    assert totals == sorted(totals)
    assert all(row["fits_budget"] is not None for row in rows)


def test_planner_rejects_context_beyond_model_limit():
    with pytest.raises(ValueError, match="cannot exceed"):
        estimate_inference_memory(_config(), context_length=8192)
