"""Autoregressive generation for the native encoder-decoder Transformer."""
from __future__ import annotations

import torch


@torch.no_grad()
def generate_seq2seq(model, tokenizer, source: str, *, max_new_tokens: int = 128, device=None) -> str:
    device = torch.device(device or next(model.parameters()).device)
    model.eval()
    bos = tokenizer.special_tokens.get("<|bos|>")
    eos = tokenizer.special_tokens.get("<|eos|>")
    if bos is None:
        raise ValueError("seq2seq generation requires <|bos|> in tokenizer")
    src_ids = tokenizer.encode(source, add_bos=True, add_eos=True)
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask = torch.ones_like(src, dtype=torch.bool)
    generated = [int(bos)]
    for _ in range(int(max_new_tokens)):
        tgt = torch.tensor([generated], dtype=torch.long, device=device)
        tgt_mask = torch.ones_like(tgt, dtype=torch.bool)
        logits = model(src, tgt, source_mask=src_mask, target_mask=tgt_mask)
        next_id = int(logits[0, -1].argmax())
        generated.append(next_id)
        if eos is not None and next_id == int(eos):
            break
    return tokenizer.decode(generated[1:], skip_special_tokens=True)
