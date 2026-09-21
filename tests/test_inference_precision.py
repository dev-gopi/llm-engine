import pytest
import torch

from inference.paged_kv_cache import PagedKVCache
from inference.quantization import (
    Q1_0_EFFECTIVE_BITS,
    dequantize_int4,
    dequantize_q1_0,
    prepare_model_for_inference,
    quantize_int4,
    quantize_q1_0,
)
from model.gpt import MiniGPT


def _model():
    return MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)


def test_bfloat16_inference_precision_is_configurable():
    model = prepare_model_for_inference(
        _model(), device=torch.device("cpu"), weight_dtype="bfloat16"
    )
    assert next(model.parameters()).dtype == torch.bfloat16
    assert model(torch.tensor([[1, 2]])).dtype == torch.bfloat16


@pytest.mark.filterwarnings(
    "ignore:torch\\.ao\\.quantization is deprecated:DeprecationWarning"
)
@pytest.mark.filterwarnings(
    "ignore:torch\\.quantize_per_tensor.*are deprecated:UserWarning"
)
def test_dynamic_int8_quantization_is_configurable():
    model = prepare_model_for_inference(
        _model(), device=torch.device("cpu"), quantization="int8_dynamic"
    )
    assert model(torch.tensor([[1, 2]])).shape == (1, 2, 16)


@pytest.mark.parametrize(
    ("dtype", "quantization"),
    [
        ("int4", "none"),
        ("float32", "int4"),
        ("float16", "none"),
        ("bfloat16", "int8_dynamic"),
    ],
)
def test_invalid_or_unsupported_cpu_precision_fails(dtype, quantization):
    with pytest.raises(ValueError):
        prepare_model_for_inference(
            _model(), device=torch.device("cpu"), weight_dtype=dtype,
            quantization=quantization,
        )


def test_invalid_gpu_quantization_fails_before_mutating_model():
    model = _model()

    with pytest.raises(ValueError, match="requires device: cpu"):
        prepare_model_for_inference(
            model,
            device=torch.device("cuda"),
            quantization="int8_dynamic",
        )

    assert next(model.parameters()).device.type == "cpu"
    assert next(model.parameters()).dtype == torch.float32


def test_portable_int4_round_trip_has_bounded_error():
    values = torch.tensor([[-1.0, -0.2, 0.0], [0.25, 0.8, 1.0]])
    packed, scale, original_numel = quantize_int4(values)
    restored = dequantize_int4(packed, scale, shape=values.shape, original_numel=original_numel)
    torch.testing.assert_close(restored, values, atol=0.15, rtol=0)


def test_q1_0_binary_pack_is_1_125_bpw_and_self_consistent():
    values = torch.linspace(-2.0, 2.0, 257).reshape(1, 257)
    packed, scales, original_numel = quantize_q1_0(values)
    restored = dequantize_q1_0(
        packed, scales, shape=values.shape, original_numel=original_numel
    )
    assert Q1_0_EFFECTIVE_BITS == 1.125
    assert packed.shape == (3, 16)
    assert scales.shape == (3,)
    assert restored.shape == values.shape
    # Binary quantization is intentionally aggressive. Validate representation
    # semantics rather than pretending it is a near-lossless round-trip.
    assert torch.all(torch.isfinite(restored))
    assert torch.all(restored.abs().sum(dim=-1) > 0)


def test_int8_paged_kv_cache_reduces_storage_and_preserves_values():
    native = PagedKVCache(
        num_pages=2, page_size=2, layers=1, kv_heads=1, head_dim=8, device="cpu", dtype=torch.float32,
    )
    compressed = PagedKVCache(
        num_pages=2, page_size=2, layers=1, kv_heads=1, head_dim=8, device="cpu", dtype=torch.float32,
        quantization="int8",
    )
    keys, values = torch.randn(1, 1, 2, 8), torch.randn(1, 1, 2, 8)
    for cache in (native, compressed):
        cache.reserve("test", 2)
        cache.append("test", keys, values)
    actual_keys, actual_values = compressed.materialize("test")
    assert compressed.storage_nbytes < native.storage_nbytes
    torch.testing.assert_close(actual_keys, keys, atol=0.02, rtol=0.02)
    torch.testing.assert_close(actual_values, values, atol=0.02, rtol=0.02)
