"""Train the native encoder-decoder Transformer on JSONL source/target pairs."""

from __future__ import annotations

try:
    from scripts._bootstrap import PROJECT_ROOT
except ModuleNotFoundError:
    from _bootstrap import PROJECT_ROOT  # noqa: F401
import argparse
import atexit
from pathlib import Path

import torch

from model.seq2seq import EncoderDecoderTransformer
from tokenizer.encoder import Tokenizer
from training.experiment_tracking import create_tracker
from training.seq2seq import (
    Seq2SeqTrainer,
    build_seq2seq_loader,
    load_seq2seq_checkpoint,
    save_seq2seq_checkpoint,
)
from utils.config import load_yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("checkpoints/seq2seq/latest.pt"))
    p.add_argument("--resume", type=Path)
    a = p.parse_args()
    cfg = load_yaml(a.config)
    tok = Tokenizer.load(a.tokenizer)
    mcfg = cfg.get("model", {})
    model = EncoderDecoderTransformer(
        vocab_size=tok.vocab_size,
        d_model=int(mcfg.get("d_model", 256)),
        nhead=int(mcfg.get("nhead", 8)),
        layers=int(mcfg.get("layers", 4)),
        ff_dim=int(mcfg.get("ff_dim", 1024)),
        max_position=int(mcfg.get("max_position", 2048)),
        dropout=float(mcfg.get("dropout", 0.0)),
    )
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu"
        else "cpu"
    )
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )
    tcfg = cfg.get("experiment_tracking") or {"kind": "none"}
    tracker = create_tracker(
        str(tcfg.get("kind", "none")),
        **{k: v for k, v in tcfg.items() if k != "kind"},
    )
    atexit.register(tracker.close)
    trainer = Seq2SeqTrainer(
        model,
        opt,
        device=device,
        gradient_clip_norm=float(cfg.get("gradient_clip_norm", 1.0)),
        tracker=tracker,
    )
    if a.resume:
        state = load_seq2seq_checkpoint(a.resume, model, opt, map_location=device)
        trainer.global_step = int(state.get("step", 0))
    train = build_seq2seq_loader(cfg["train_files"], tok, cfg, shuffle=True)
    valid = (
        build_seq2seq_loader(
            cfg["validation_files"],
            tok,
            {
                **cfg,
                "batch_size": int(
                    cfg.get("validation_batch_size", cfg.get("batch_size", 8))
                ),
            },
            shuffle=False,
        )
        if cfg.get("validation_files")
        else None
    )
    meta = {"config": cfg, "tokenizer_fingerprint": tok.fingerprint}

    def save(cur, epoch):
        save_seq2seq_checkpoint(
            a.output,
            model,
            opt,
            step=cur.global_step,
            metadata={**meta, "epoch": epoch + 1},
        )

    def save_best(cur, epoch):
        save_seq2seq_checkpoint(
            a.output.with_name("best.pt"),
            model,
            opt,
            step=cur.global_step,
            metadata={**meta, "epoch": epoch + 1, "best": True},
        )

    trainer.fit(
        train,
        epochs=int(cfg.get("epochs", 1)),
        validation_loader=valid,
        checkpoint_callback=save,
        best_checkpoint_callback=save_best,
    )


if __name__ == "__main__":
    main()
