"""Export a native seq2seq checkpoint to safetensors plus self-describing metadata."""

from __future__ import annotations

try:
    from scripts._bootstrap import PROJECT_ROOT
except ModuleNotFoundError:
    from _bootstrap import PROJECT_ROOT  # noqa: F401
import argparse
import json
from pathlib import Path

from safetensors.torch import save_model

from model.seq2seq import EncoderDecoderTransformer
from tokenizer.encoder import Tokenizer
from training.seq2seq import load_seq2seq_checkpoint
from utils.config import load_yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
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
    state = load_seq2seq_checkpoint(a.checkpoint, model, map_location="cpu")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    save_model(model, str(a.output_dir / "model.safetensors"))
    metadata = {
        "architecture": "EncoderDecoderTransformer",
        "model": m,
        "vocab_size": tok.vocab_size,
        "tokenizer_fingerprint": tok.fingerprint,
        "checkpoint_metadata": state.get("metadata", {}),
    }
    (a.output_dir / "config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(a.output_dir)


if __name__ == "__main__":
    main()
