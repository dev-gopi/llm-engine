import pytest
import torch

from model.gpt import MiniGPT
from model.positional import PositionalEmbedding


def test_default_positions_and_shape() -> None:
    module = PositionalEmbedding(8, 4)
    tokens = torch.zeros((2, 3), dtype=torch.long)
    expected = module.weight[:3].unsqueeze(0).expand(2, -1, -1)
    assert torch.equal(module(tokens), expected)


def test_explicit_positions_broadcast_and_offset() -> None:
    module = PositionalEmbedding(8, 4)
    tokens = torch.zeros((2, 2), dtype=torch.long)
    expected = module.weight[3:5].unsqueeze(0).expand(2, -1, -1)
    assert torch.equal(module(tokens, position_offset=3), expected)
    ids = torch.tensor([4, 1])
    expected = module.weight[ids].unsqueeze(0).expand(2, -1, -1)
    assert torch.equal(module(tokens, position_ids=ids), expected)


def test_padding_uses_compact_positions_and_returns_zero_vectors() -> None:
    module = PositionalEmbedding(8, 3)
    tokens = torch.zeros((2, 4), dtype=torch.long)
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 0]])
    output = module(tokens, attention_mask=mask)
    assert torch.count_nonzero(output[0, :2]) == 0
    assert torch.count_nonzero(output[1, 3]) == 0
    assert torch.equal(output[0, 2:], module.weight[:2])
    assert torch.equal(output[1, :3], module.weight[:3])


def test_bounds_and_input_validation() -> None:
    module = PositionalEmbedding(4, 2)
    tokens = torch.zeros((1, 2), dtype=torch.long)
    with pytest.raises(IndexError, match="position IDs"):
        module(tokens, position_offset=3)
    with pytest.raises(ValueError, match="non-negative"):
        module(tokens, position_offset=-1)
    with pytest.raises(TypeError, match="integer dtype"):
        module(tokens, position_ids=torch.tensor([0.0, 1.0]))


def test_resize_preserves_existing_weights_and_initializes_new_rows() -> None:
    module = PositionalEmbedding(3, 4, initializer_range=0.01)
    original = module.weight.detach().clone()
    module.resize(6)
    assert module.max_positions == 6
    assert torch.equal(module.weight[:3], original)
    assert module.weight.shape == (6, 4)
    assert torch.count_nonzero(module.weight[3:]) > 0


def test_from_config_and_gradient_flow() -> None:
    module = PositionalEmbedding.from_config(
        {"max_position": 5, "hidden_size": 3, "initializer_range": 0.01}
    )
    module(torch.zeros((2, 4), dtype=torch.long)).sum().backward()
    assert module.weight.grad is not None
    assert module.weight.shape == (5, 3)


def test_gpt_accepts_explicit_position_controls() -> None:
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8).eval()
    tokens = torch.tensor([[1, 2]])
    with torch.no_grad():
        offset_logits = model(tokens, position_offset=2)
        explicit_logits = model(tokens, position_ids=torch.tensor([2, 3]))
    assert torch.allclose(offset_logits, explicit_logits)
