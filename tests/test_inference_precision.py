import pytest
import torch

from inference.quantization import prepare_model_for_inference
from model.gpt import MiniGPT


def _model():
    return MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)


def test_bfloat16_inference_precision_is_configurable():
    model = prepare_model_for_inference(
        _model(), device=torch.device("cpu"), weight_dtype="bfloat16"
    )
    assert next(model.parameters()).dtype == torch.bfloat16
    assert model(torch.tensor([[1, 2]])).dtype == torch.bfloat16


def test_dynamic_int8_quantization_is_configurable():
    model = prepare_model_for_inference(
        _model(), device=torch.device("cpu"), quantization="int8_dynamic"
    )
    assert model(torch.tensor([[1, 2]])).shape == (1, 2, 16)


@pytest.mark.parametrize(
    ("dtype", "quantization"),
    [("int4", "none"), ("float32", "int4"), ("float16", "none")],
)
def test_invalid_or_unsupported_cpu_precision_fails(dtype, quantization):
    with pytest.raises(ValueError):
        prepare_model_for_inference(
            _model(), device=torch.device("cpu"), weight_dtype=dtype,
            quantization=quantization,
        )
