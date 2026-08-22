import pytest
import torch
import torch.nn.functional as F

from model.feed_forward import FeedForward


def test_default_shape_and_hidden_expansion():
    module = FeedForward(dim=32)
    hidden_states = torch.randn(2, 7, 32)
    assert module.hidden_dim == 128
    assert module(hidden_states).shape == hidden_states.shape


def test_standard_gelu_matches_manual_reference():
    torch.manual_seed(20)
    module = FeedForward(dim=16, hidden_dim=40, activation="gelu", dropout=0.0).eval()
    hidden_states = torch.randn(2, 5, 16)
    expected = F.linear(
        F.gelu(F.linear(hidden_states, module.in_proj.weight, module.in_proj.bias)),
        module.out_proj.weight,
        module.out_proj.bias,
    )
    torch.testing.assert_close(module(hidden_states), expected)


@pytest.mark.parametrize(
    ("activation", "gate_function"),
    [("swiglu", F.silu), ("geglu", F.gelu)],
)
def test_gated_variants_match_manual_reference(activation, gate_function):
    torch.manual_seed(21)
    module = FeedForward(dim=16, hidden_dim=32, activation=activation).eval()
    hidden_states = torch.randn(2, 5, 16)
    gate, value = module.in_proj(hidden_states).chunk(2, dim=-1)
    expected = module.out_proj(gate_function(gate) * value)
    assert module.in_proj.out_features == 64
    torch.testing.assert_close(module(hidden_states), expected)


@pytest.mark.parametrize("activation", ["gelu", "gelu_tanh", "relu", "silu"])
def test_standard_activations_are_finite(activation):
    module = FeedForward(dim=8, activation=activation)
    output = module(torch.randn(2, 3, 8))
    assert torch.isfinite(output).all()


def test_hidden_size_rounds_up_for_hardware_alignment():
    module = FeedForward(dim=10, expansion_factor=3.1, multiple_of=16)
    assert module.hidden_dim == 32
    assert module.in_proj.out_features == 32


def test_explicit_hidden_size_takes_precedence_and_is_aligned():
    module = FeedForward(dim=16, hidden_dim=70, expansion_factor=10, multiple_of=32)
    assert module.hidden_dim == 96


def test_initialization_and_zero_biases():
    torch.manual_seed(22)
    module = FeedForward(dim=128, hidden_dim=512, initializer_range=0.01)
    assert module.in_proj.weight.std().item() == pytest.approx(0.01, rel=0.03)
    assert module.out_proj.weight.std().item() == pytest.approx(0.01, rel=0.03)
    assert module.in_proj.bias is not None
    assert module.out_proj.bias is not None
    assert module.in_proj.bias.count_nonzero().item() == 0
    assert module.out_proj.bias.count_nonzero().item() == 0


def test_bias_can_be_disabled():
    module = FeedForward(dim=16, bias=False)
    assert module.in_proj.bias is None
    assert module.out_proj.bias is None


def test_dropout_only_changes_training_outputs():
    module = FeedForward(dim=16, dropout=0.5)
    hidden_states = torch.randn(4, 8, 16)
    module.eval()
    torch.testing.assert_close(module(hidden_states), module(hidden_states))
    module.train()
    torch.manual_seed(1)
    first = module(hidden_states)
    torch.manual_seed(2)
    second = module(hidden_states)
    assert not torch.equal(first, second)


def test_backward_produces_finite_gradients():
    module = FeedForward(dim=32, activation="swiglu")
    hidden_states = torch.randn(2, 8, 32, requires_grad=True)
    module(hidden_states).square().mean().backward()
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    assert all(parameter.grad is not None for parameter in module.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in module.parameters())


def test_dtype_and_two_dimensional_input():
    module = FeedForward(dim=12, dtype=torch.float64)
    hidden_states = torch.randn(5, 12, dtype=torch.float64)
    output = module(hidden_states)
    assert output.shape == hidden_states.shape
    assert output.dtype == torch.float64


def test_state_dict_round_trip():
    original = FeedForward(dim=16, hidden_dim=32, activation="swiglu")
    restored = FeedForward(dim=16, hidden_dim=32, activation="swiglu")
    restored.load_state_dict(original.state_dict())
    hidden_states = torch.randn(2, 4, 16)
    torch.testing.assert_close(restored(hidden_states), original(hidden_states))


def test_from_model_config():
    module = FeedForward.from_config(
        {
            "hidden_size": 64,
            "ffn_hidden_size": 250,
            "ffn_multiple_of": 64,
            "ffn_activation": "swiglu",
            "ffn_dropout": 0.1,
            "ffn_bias": False,
            "initializer_range": 0.01,
        }
    )
    assert module.dim == 64
    assert module.hidden_dim == 256
    assert module.activation_name == "swiglu"
    assert module.dropout_probability == 0.1
    assert module.in_proj.bias is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dim": 0},
        {"dim": 16, "hidden_dim": 0},
        {"dim": 16, "expansion_factor": 0},
        {"dim": 16, "multiple_of": 0},
        {"dim": 16, "activation": "unknown"},
        {"dim": 16, "dropout": 1.0},
        {"dim": 16, "initializer_range": 0.0},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises((TypeError, ValueError)):
        FeedForward(**kwargs)


def test_invalid_hidden_states():
    module = FeedForward(dim=16)
    with pytest.raises(TypeError, match="torch.Tensor"):
        module([[1.0] * 16])
    with pytest.raises(ValueError, match="at least two"):
        module(torch.randn(16))
    with pytest.raises(ValueError, match="final dimension"):
        module(torch.randn(2, 4, 15))
    with pytest.raises(TypeError, match="floating-point"):
        module(torch.ones(2, 4, 16, dtype=torch.long))
