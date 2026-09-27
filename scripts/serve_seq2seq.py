"""Run one-shot native seq2seq generation from a trained checkpoint."""

from __future__ import annotations

try:
    from scripts._bootstrap import PROJECT_ROOT
except ModuleNotFoundError:
    from _bootstrap import PROJECT_ROOT  # noqa: F401
import argparse
from pathlib import Path

import torch

from inference.seq2seq import generate_seq2seq
from model.seq2seq import EncoderDecoderTransformer
from tokenizer.encoder import Tokenizer
from training.seq2seq import load_seq2seq_checkpoint
from utils.config import load_yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--max-new-tokens", type=int, default=128)
    a = p.parse_args()
    cfg = load_yaml(a.config)
    tok = Tokenizer.load(a.tokenizer)
    m = cfg.get("model", {})
    model = EncoderDecoderTransformer(
        tok.vocab_size,
        int(m.get("d_model", 256)),
        int(m.get("nhead", 8)),
        int(m.get("layers", 4)),
        int(m.get("ff_dim", 1024)),
        int(m.get("max_position", 2048)),
        float(m.get("dropout", 0.0)),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    load_seq2seq_checkpoint(a.checkpoint, model, map_location=device)
    print(
        generate_seq2seq(
            model, tok, a.text, max_new_tokens=a.max_new_tokens, device=device
        )
    )


if __name__ == "__main__":
    main()
