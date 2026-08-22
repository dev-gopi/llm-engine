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
