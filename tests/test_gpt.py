import pytest
import torch

from model.gpt import MiniGPT
from utils.config import load_yaml


def test_gpt_embedding_integration_forward_and_backward():
    model = MiniGPT(vocab_size=64, dim=16, layers=2, heads=4, max_pos=16)
    token_ids = torch.randint(0, 64, (2, 8))
    logits = model(token_ids)
    assert logits.shape == (2, 8, 64)
    logits.mean().backward()
    assert model.tok.weight.grad is not None
    assert torch.isfinite(model.tok.weight.grad).all()


def test_gpt_accepts_padding_attention_mask():
    model = MiniGPT(vocab_size=64, dim=16, layers=2, heads=4, max_pos=16).eval()
    token_ids = torch.randint(0, 64, (2, 8))
    attention_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]]
    )
    logits = model(token_ids, attention_mask=attention_mask)
    assert logits.shape == (2, 8, 64)


def test_gpt_logits_are_causal():
    torch.manual_seed(12)
    model = MiniGPT(vocab_size=64, dim=16, layers=2, heads=4, max_pos=16).eval()
    original = torch.randint(0, 64, (1, 8))
    modified = original.clone()
    modified[:, 5:] = torch.randint(0, 64, (1, 3))
    torch.testing.assert_close(model(original)[:, :5], model(modified)[:, :5])


def test_gpt_ties_embedding_and_lm_head_weights():
    model = MiniGPT(vocab_size=32, dim=8, layers=1, heads=2)
    assert model.head.weight is model.tok.weight
    assert model.head.bias is None


def test_gpt_builds_all_layers_from_config():
    config = load_yaml("configs/model.yaml")
    config.update({"vocab_size": 64, "hidden_size": 16, "layers": 2, "heads": 4,
                   "max_position": 32, "ffn_hidden_size": 32, "ffn_multiple_of": 1})
    model = MiniGPT.from_config(config)
    assert len(model.blocks) == 2
    assert model.blocks[0].ffn.hidden_dim == 32
    assert model.norm.dim == 16
    assert model(torch.ones((1, 4), dtype=torch.long)).shape == (1, 4, 64)


def test_gpt_resizes_tied_vocabulary():
    model = MiniGPT(vocab_size=32, dim=8, layers=1, heads=2)
    original = model.tok.weight.detach().clone()
    assert model.resize_token_embeddings(35, pad_to_multiple_of=8) == 40
    assert model.head.out_features == 40
    assert model.head.weight is model.tok.weight
    torch.testing.assert_close(model.tok.weight[:32], original)


def test_gpt_validates_inputs_and_configuration():
    with pytest.raises(ValueError, match="divisible"):
        MiniGPT(vocab_size=32, dim=10, layers=1, heads=4)
    model = MiniGPT(vocab_size=32, dim=8, layers=1, heads=2)
    with pytest.raises(TypeError, match="integer dtype"):
        model(torch.ones((1, 3)))


def test_gpt_kv_cache_matches_full_sequence_logits():
    torch.manual_seed(9)
    model = MiniGPT(vocab_size=32, dim=8, layers=2, heads=2, max_pos=8).eval()
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        full = model(tokens)
        prefix_logits, cache = model(tokens[:, :3], use_cache=True)
        step_logits, updated = model(tokens[:, 3:], past_key_values=cache, use_cache=True)
    assert prefix_logits.shape == (1, 3, 32)
    torch.testing.assert_close(step_logits[:, -1], full[:, -1], atol=1e-5, rtol=1e-5)
    assert updated[0][0].shape[2] == 4
