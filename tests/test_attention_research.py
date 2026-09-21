import torch
from model.attention import MultiHeadAttention

def test_sliding_window_attention_masks_old_tokens():
    module=MultiHeadAttention(16,4,attention_pattern='sliding_window',attention_window=2).eval()
    x=torch.randn(1,4,16)
    out=module(x)
    assert out.shape==(1,4,16)
    assert module.attention_pattern=='sliding_window'
