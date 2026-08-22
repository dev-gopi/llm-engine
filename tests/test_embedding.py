import math

import pytest
import torch
import torch.nn as nn

from model.embedding import TokenEmbedding


def test_output_shape_dtype_and_device():
    module = TokenEmbedding(100, 32, dtype=torch.float64)
    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    output = module(token_ids)
    assert output.shape == (2, 3, 32)
    assert output.dtype == torch.float64
    assert output.device == token_ids.device


def test_gpt_initialization_and_padding_row():
    torch.manual_seed(7)
    module = TokenEmbedding(10_000, 16, padding_idx=0, initializer_range=0.02)
    non_padding = module.weight[1:]
    assert module.weight[0].count_nonzero().item() == 0
    assert abs(non_padding.mean().item()) < 0.002
    assert non_padding.std().item() == pytest.approx(0.02, rel=0.03)


def test_padding_row_has_no_gradient():
    module = TokenEmbedding(20, 8, padding_idx=0)
    module(torch.tensor([[0, 1, 2, 0]])).sum().backward()
    assert module.weight.grad is not None
    assert module.weight.grad[0].count_nonzero().item() == 0
    assert module.weight.grad[1].count_nonzero().item() == 8


def test_optional_embedding_scaling():
    plain = TokenEmbedding(20, 8)
    scaled = TokenEmbedding(20, 8, scale_embeddings=True)
    with torch.no_grad():
        scaled.weight.copy_(plain.weight)
    token_ids = torch.tensor([1, 2, 3])
    torch.testing.assert_close(scaled(token_ids), plain(token_ids) * math.sqrt(8))


def test_resize_grows_preserves_weights_and_rounds_for_hardware():
    module = TokenEmbedding(10, 4, padding_idx=0)
    original = module.weight.detach().clone()
    effective_size = module.resize(17, pad_to_multiple_of=8)
    assert effective_size == 24
    assert module.vocab_size == 24
    torch.testing.assert_close(module.weight[:10], original)
    assert module.weight[0].count_nonzero().item() == 0
    assert module.weight[10:].std().item() > 0


def test_resize_shrinks_and_preserves_freeze_state():
    module = TokenEmbedding(12, 4).freeze()
    original = module.weight.detach().clone()
    assert module.resize(8) == 8
    torch.testing.assert_close(module.weight, original[:8])
    assert not module.weight.requires_grad
    assert module.unfreeze().weight.requires_grad


def test_weight_tying_shares_parameter_and_gradient():
    module = TokenEmbedding(32, 12)
    projection = nn.Linear(12, 32, bias=False)
    module.tie_weights(projection)
    assert projection.weight is module.weight
    hidden = module(torch.tensor([[1, 2, 3]]))
    projection(hidden).sum().backward()
    assert module.weight.grad is projection.weight.grad
    assert module.weight.grad is not None


def test_state_dict_round_trip():
    original = TokenEmbedding(32, 12, padding_idx=0)
    restored = TokenEmbedding(32, 12, padding_idx=0)
    restored.load_state_dict(original.state_dict())
    token_ids = torch.tensor([[0, 4, 7]])
    torch.testing.assert_close(restored(token_ids), original(token_ids))


def test_from_model_config():
    module = TokenEmbedding.from_config(
        {
            "vocab_size": 128,
            "hidden_size": 24,
            "padding_idx": 0,
            "initializer_range": 0.01,
            "scale_embeddings": True,
            "freeze_embeddings": True,
        }
    )
    assert module.vocab_size == 128
    assert module.embedding_dim == 24
    assert module.padding_idx == 0
    assert module.initializer_range == 0.01
    assert module.scale_embeddings
    assert not module.weight.requires_grad


@pytest.mark.parametrize(
    ("arguments", "error_type"),
    [
        ((0, 8), ValueError),
        ((10, 0), ValueError),
        ((10.0, 8), TypeError),
    ],
)
def test_invalid_dimensions(arguments, error_type):
    with pytest.raises(error_type):
        TokenEmbedding(*arguments)


def test_invalid_padding_and_initializer():
    with pytest.raises(ValueError, match="padding_idx"):
        TokenEmbedding(10, 8, padding_idx=10)
    with pytest.raises(ValueError, match="initializer_range"):
        TokenEmbedding(10, 8, initializer_range=float("nan"))


def test_forward_requires_integer_tensor():
    module = TokenEmbedding(10, 8)
    with pytest.raises(TypeError, match="torch.Tensor"):
        module([1, 2, 3])
    with pytest.raises(TypeError, match="int32 or torch.int64"):
        module(torch.tensor([1.0, 2.0]))


def test_weight_tying_validates_projection_shape():
    module = TokenEmbedding(32, 12)
    with pytest.raises(ValueError, match="shape"):
        module.tie_weights(nn.Linear(12, 31, bias=False))
    with pytest.raises(TypeError, match="Linear"):
        module.tie_weights(nn.Identity())
