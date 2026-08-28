import pytest
import torch
import torch.nn.functional as F

from model.layer_norm import LayerNorm, RMSNorm, build_normalization, normalization_from_config


def test_layer_norm_matches_pytorch_reference():
    torch.manual_seed(30)
    module = LayerNorm(16, eps=1e-5)
    with torch.no_grad():
        module.weight.uniform_(0.5, 1.5)
        module.bias.uniform_(-0.2, 0.2)
    hidden_states = torch.randn(2, 5, 16)
    expected = F.layer_norm(
        hidden_states, (16,), module.weight, module.bias, module.eps
    )
    torch.testing.assert_close(module(hidden_states), expected)


def test_layer_norm_zero_mean_and_unit_variance():
    module = LayerNorm(64, eps=1e-7, bias=False)
    output = module(torch.randn(4, 8, 64) * 5 + 10)
    torch.testing.assert_close(output.mean(dim=-1), torch.zeros(4, 8), atol=1e-6, rtol=0)
    torch.testing.assert_close(output.var(dim=-1, unbiased=False), torch.ones(4, 8), atol=1e-5, rtol=0)


def test_rms_norm_matches_manual_reference():
    torch.manual_seed(31)
    module = RMSNorm(16, eps=1e-6, bias=True)
    with torch.no_grad():
        module.weight.uniform_(0.5, 1.5)
        module.bias.uniform_(-0.2, 0.2)
    hidden_states = torch.randn(2, 5, 16)
    expected = (
        hidden_states
        * torch.rsqrt(hidden_states.square().mean(dim=-1, keepdim=True) + module.eps)
        * module.weight
        + module.bias
    )
    torch.testing.assert_close(module(hidden_states), expected)


def test_rms_norm_low_precision_is_finite():
    module = RMSNorm(32, dtype=torch.bfloat16)
    hidden_states = torch.randn(2, 4, 32, dtype=torch.bfloat16) * 100
    output = module(hidden_states)
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rms_norm_preserves_low_precision_input_dtype_with_fp32_parameters(dtype):
    module = RMSNorm(8)
    output = module(torch.randn(2, 3, 8, dtype=dtype))
    assert output.dtype == dtype


def test_bias_can_be_disabled_and_parameters_reset():
    module = LayerNorm(8, bias=False)
    assert module.bias is None
    with torch.no_grad():
        module.weight.zero_()
    module.reset_parameters()
    torch.testing.assert_close(module.weight, torch.ones(8))


def test_normalization_factory_and_config():
    assert isinstance(build_normalization("layer_norm", 8), LayerNorm)
    assert isinstance(build_normalization("rmsnorm", 8), RMSNorm)
    configured = normalization_from_config(
        {
            "hidden_size": 12,
            "norm_type": "rms_norm",
            "norm_eps": 1e-6,
            "norm_bias": False,
        }
    )
    assert isinstance(configured, RMSNorm)
    assert configured.dim == 12
    assert configured.eps == 1e-6
    assert configured.bias is None


def test_backward_and_state_dict_round_trip():
    original = LayerNorm(16)
    hidden_states = torch.randn(2, 4, 16, requires_grad=True)
    original(hidden_states).square().mean().backward()
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()

    restored = LayerNorm(16)
    restored.load_state_dict(original.state_dict())
    sample = torch.randn(2, 4, 16)
    torch.testing.assert_close(restored(sample), original(sample))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: LayerNorm(0),
        lambda: RMSNorm(0),
        lambda: LayerNorm(8, eps=0),
        lambda: RMSNorm(8, eps=1),
        lambda: build_normalization("unknown", 8),
    ],
)
def test_invalid_configuration(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_invalid_hidden_states():
    module = LayerNorm(8)
    with pytest.raises(TypeError, match="torch.Tensor"):
        module([[0.0] * 8])
    with pytest.raises(ValueError, match="at least two"):
        module(torch.randn(8))
    with pytest.raises(ValueError, match="final dimension"):
        module(torch.randn(2, 7))
    with pytest.raises(TypeError, match="floating-point"):
        module(torch.ones(2, 8, dtype=torch.long))
