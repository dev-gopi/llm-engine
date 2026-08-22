import torch

from model.gpt import MiniGPT


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
