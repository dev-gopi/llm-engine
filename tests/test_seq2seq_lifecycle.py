import torch

from model.seq2seq import EncoderDecoderTransformer
from training.seq2seq import (
    Seq2SeqTrainer,
    load_seq2seq_checkpoint,
    save_seq2seq_checkpoint,
)


def test_seq2seq_trainer_checkpoint_roundtrip(tmp_path):
    model = EncoderDecoderTransformer(32, d_model=16, nhead=4, layers=1, ff_dim=32, max_position=16)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Seq2SeqTrainer(model, opt, device="cpu")
    batch = {
        "source_ids": torch.randint(0, 32, (2, 5)),
        "target_ids": torch.randint(0, 32, (2, 6)),
        "source_mask": torch.ones(2, 5, dtype=torch.bool),
        "target_mask": torch.ones(2, 6, dtype=torch.bool),
    }
    metrics = trainer.train_step(batch)
    assert metrics["train/loss"] > 0
    path = tmp_path / "s.pt"
    save_seq2seq_checkpoint(path, model, opt, step=trainer.global_step, metadata={"x": 1})
    clone = EncoderDecoderTransformer(32, d_model=16, nhead=4, layers=1, ff_dim=32, max_position=16)
    state = load_seq2seq_checkpoint(path, clone, map_location="cpu")
    assert state["step"] == 1 and state["metadata"]["x"] == 1
    for a, b in zip(model.parameters(), clone.parameters()):
        assert torch.equal(a, b)
