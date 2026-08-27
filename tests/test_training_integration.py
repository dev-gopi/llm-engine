import torch
import pytest

from datasets.collator import Collator
from model.gpt import MiniGPT
from optim.adamw import build_adamw
from optim.ema import EMA
from optim.scheduler import Scheduler
from training.trainer import Trainer


def test_dataset_batch_trains_model_optimizer_scheduler_and_ema() -> None:
    model = MiniGPT(vocab_size=32, dim=8, layers=1, heads=2, max_pos=8)
    batch = Collator(pad_token_id=0)([
        torch.tensor([1, 2, 3, 4]),
        torch.tensor([5, 6, 7]),
    ])
    optimizer = build_adamw(model, learning_rate=1e-3)
    scheduler = Scheduler(optimizer, warmup_steps=0, total_steps=3)
    ema = EMA(model, decay=0.9)
    trainer = Trainer(model, optimizer, scheduler=scheduler, ema=ema)
    loss = trainer.train_step(batch)
    assert loss > 0
    assert ema.num_updates == 1
    assert scheduler.last_epoch == 1


def test_gradient_accumulation_steps_optimizer_once() -> None:
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    optimizer = build_adamw(model, learning_rate=1e-3)
    trainer = Trainer(model, optimizer, gradient_accumulation_steps=2)
    batch = Collator(0)([torch.tensor([1, 2, 3])])
    before = model.tok.weight.detach().clone()
    trainer.train_step(batch)
    assert trainer.global_step == 0
    torch.testing.assert_close(model.tok.weight, before)
    trainer.train_step(batch)
    assert trainer.global_step == 1
    assert not torch.equal(model.tok.weight, before)


def test_trainer_tracks_observability_metrics() -> None:
    model = MiniGPT(vocab_size=32, dim=8, layers=1, heads=2, max_pos=8)
    optimizer = build_adamw(model, learning_rate=1e-3)
    trainer = Trainer(model, optimizer)
    batch = Collator(0)([
        torch.tensor([1, 2, 3, 4]),
        torch.tensor([5, 6, 7]),
    ])

    trainer.train_step(batch)

    assert trainer.tokens_processed == 5
    assert trainer.training_seconds > 0
    assert trainer.tokens_per_second > 0
    assert trainer.learning_rate == pytest.approx(1e-3)
    assert torch.isfinite(torch.tensor(trainer.last_gradient_norm))
    assert trainer.last_gradient_norm > 0
    assert trainer.peak_memory_mb == 0
    assert trainer.nonfinite_updates == 0


def test_trainer_rejects_nonfinite_loss_before_updating_weights() -> None:
    class NonfiniteLoss:
        def __call__(self, logits, _targets, *, loss_mask=None):
            return logits.sum() * torch.tensor(float("nan"), device=logits.device)

    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    optimizer = build_adamw(model, learning_rate=1e-3)
    trainer = Trainer(model, optimizer, loss_fn=NonfiniteLoss())
    batch = Collator(0)([torch.tensor([1, 2, 3])])
    before = model.tok.weight.detach().clone()

    with pytest.raises(FloatingPointError, match="non-finite training loss"):
        trainer.train_step(batch)

    assert trainer.nonfinite_updates == 1
    assert trainer.global_step == 0
    torch.testing.assert_close(model.tok.weight, before)
