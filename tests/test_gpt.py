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


@pytest.mark.parametrize("position_type", ["learned", "rotary"])
def test_binary_float_mask_excludes_padding_in_forward_and_cache(position_type):
    torch.manual_seed(12)
    model = MiniGPT(vocab_size=64, dim=16, layers=2, heads=4, kv_heads=2,
                    max_pos=16, position_type=position_type).eval()
    tokens = torch.tensor([[0, 0, 4, 5]])
    mask = torch.tensor([[0, 0, 1, 1]], dtype=torch.bool)
    expected, cache = model(tokens, attention_mask=mask, use_cache=True)
    actual, float_cache = model(tokens, attention_mask=mask.float(), use_cache=True)
    torch.testing.assert_close(actual, expected)
    changed = tokens.clone()
    changed[:, :2] = 9
    torch.testing.assert_close(model(changed, attention_mask=mask.float())[:, 2:], expected[:, 2:])
    next_mask = torch.cat([mask, torch.ones((1, 1), dtype=torch.bool)], dim=1)
    next_token = torch.tensor([[6]])
    torch.testing.assert_close(
        model(next_token, attention_mask=next_mask, past_key_values=cache),
        model(next_token, attention_mask=next_mask.float(), past_key_values=float_cache),
    )


def test_gpt_ties_embedding_and_lm_head_weights():
    model = MiniGPT(vocab_size=32, dim=8, layers=1, heads=2)
    assert model.head.weight is model.tok.weight
    assert model.head.bias is None


def test_gpt_builds_all_layers_from_config():
    config = load_yaml("configs/model.cpu.yaml")
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
    with pytest.raises(ValueError, match="binary"):
        model(torch.ones((1, 3), dtype=torch.long), attention_mask=torch.tensor([[1, 2, 1]]))
    with pytest.raises(ValueError, match="non-negative"):
        model(torch.ones((1, 3), dtype=torch.long), position_offset=-1)


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


@pytest.mark.parametrize("position_type", ["learned", "rotary"])
def test_gpt_cached_padding_mask_matches_full_sequence(position_type):
    torch.manual_seed(19)
    model = MiniGPT(
        vocab_size=32, dim=8, layers=2, heads=2, max_pos=8, position_type=position_type
    ).eval()
    tokens = torch.tensor([[0, 0, 3, 4]])
    mask = torch.tensor([[0, 0, 1, 1]])
    with torch.no_grad():
        full = model(tokens, attention_mask=mask)
        _, cache = model(tokens[:, :3], attention_mask=mask[:, :3], use_cache=True)
        step = model(tokens[:, 3:], attention_mask=mask, past_key_values=cache)
    torch.testing.assert_close(step[:, -1], full[:, -1], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("position_type", ["learned", "rotary"])
def test_last_token_projection_preserves_logits_and_full_cache(position_type):
    model = MiniGPT(vocab_size=32, dim=16, layers=2, heads=4,
                    max_pos=16, position_type=position_type).eval()
    ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    with torch.no_grad():
        full, full_cache = model(ids, use_cache=True)
        last, last_cache = model(ids, use_cache=True, logits_to_keep=1)
    assert last.shape == (2, 1, 32)
    torch.testing.assert_close(last, full[:, -1:])
    for expected, actual in zip(full_cache, last_cache):
        for x, y in zip(expected, actual):
            torch.testing.assert_close(x, y)


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_invalid_logits_to_keep(value):
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2)
    with pytest.raises((TypeError, ValueError), match="logits_to_keep"):
        model(torch.tensor([[1, 2]]), logits_to_keep=value)


@pytest.mark.parametrize("position_type", ["learned", "rotary", "sinusoidal"])
def test_padding_aware_positions_match_unpadded_prefill_and_decode(position_type):
    torch.manual_seed(95)
    model = MiniGPT(vocab_size=32, dim=16, layers=2, heads=4, kv_heads=2,
                    max_pos=16, position_type=position_type).eval()
    # Include an internal masked position, so relative distances matter for RoPE.
    padded = torch.tensor([[0, 3, 0, 4, 5]])
    mask = torch.tensor([[0, 1, 0, 1, 1]])
    plain = torch.tensor([[3, 4, 5]])
    with torch.no_grad():
        expected, plain_cache = model(plain, use_cache=True)
        actual, padded_cache = model(padded, attention_mask=mask, use_cache=True)
        torch.testing.assert_close(actual[:, [1, 3, 4]], expected)
        expected_next, _ = model(torch.tensor([[6]]), past_key_values=plain_cache, use_cache=True)
        actual_next, _ = model(torch.tensor([[6]]),
                               attention_mask=torch.tensor([[0, 1, 0, 1, 1, 1]]),
                               past_key_values=padded_cache, use_cache=True)
    torch.testing.assert_close(actual_next, expected_next)


def test_sinusoidal_model_honors_explicit_position_ids():
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2,
                    max_pos=16, position_type="sinusoidal").eval()
    tokens = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        expected = model(tokens, position_offset=4)
        actual = model(tokens, position_ids=torch.tensor([4, 5, 6]))
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("static_cache", [False, True])
def test_rotary_autocast_cached_decoding_matches_full_forward(static_cache):
    from model.kv_cache import StaticLayerKVCache
    torch.manual_seed(121)
    model = MiniGPT(vocab_size=32, dim=16, layers=2, heads=4, kv_heads=2,
                    max_pos=16, position_type="rotary").eval()
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        full = model(tokens)
        _, cache = model(tokens[:, :3], use_cache=True)
        for key, value in cache:
            assert key.dtype == value.dtype == torch.bfloat16
        if static_cache:
            cache = tuple(StaticLayerKVCache(k, v, capacity=16) for k, v in cache)
        actual, _ = model(tokens[:, 3:], past_key_values=cache, use_cache=True)
    torch.testing.assert_close(actual, full[:, -1:], atol=2e-3, rtol=2e-2)
