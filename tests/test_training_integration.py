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
    assert trainer.gpu_memory_mb == (0.0, 0.0, 0.0)
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


def test_trainer_validates_grad_scaler_configuration() -> None:
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    optimizer = build_adamw(model, learning_rate=1e-3)

    with pytest.raises(ValueError, match="grad_scaler_initial_scale"):
        Trainer(model, optimizer, grad_scaler_initial_scale=0)
    with pytest.raises(ValueError, match="grad_scaler_growth_interval"):
        Trainer(model, optimizer, grad_scaler_growth_interval=0)


def test_token_metrics_follow_configured_shift_and_ignore_index():
    from model.loss import CausalLanguageModelLoss
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    trainer = Trainer(model, build_adamw(model, learning_rate=1e-3),
                      loss_fn=CausalLanguageModelLoss(shift_labels=False, ignore_index=-1))
    trainer.train_step({"input_ids": torch.tensor([[1, 2, 3]]),
                        "labels": torch.tensor([[4, -1, 5]]),
                        "loss_mask": torch.tensor([[1, 1, 0]])})
    assert trainer.tokens_processed == 1


def test_nonfinite_loss_resets_discarded_accumulation_window():
    class ToggleLoss:
        fail = False
        def __call__(self, logits, targets, *, loss_mask=None):
            from model.loss import CausalLanguageModelLoss
            if self.fail:
                return logits.sum() * float("nan")
            return CausalLanguageModelLoss()(logits, targets)
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    objective = ToggleLoss()
    trainer = Trainer(model, build_adamw(model, learning_rate=1e-3),
                      loss_fn=objective, gradient_accumulation_steps=2)
    batch = Collator(0)([torch.tensor([1, 2, 3])])
    trainer.train_step(batch)
    objective.fail = True
    with pytest.raises(FloatingPointError):
        trainer.train_step(batch)
    assert trainer.micro_step == 0
    objective.fail = False
    trainer.train_step(batch)
    assert trainer.global_step == 0
    trainer.train_step(batch)
    assert trainer.global_step == 1


def test_checkpoint_resume_matches_uninterrupted_updates(tmp_path):
    from training.checkpoint import save_checkpoint, load_checkpoint
    torch.manual_seed(123)
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    optimizer = build_adamw(model, learning_rate=1e-3)
    scheduler = Scheduler(optimizer, warmup_steps=0, total_steps=6)
    trainer = Trainer(model, optimizer, scheduler=scheduler)
    batch = Collator(0)([torch.tensor([1, 2, 3, 4])])
    for _ in range(3):
        trainer.train_step(batch)
    path = save_checkpoint(tmp_path / 'resume.pt', model, optimizer=optimizer,
                           scheduler=scheduler, trainer=trainer.state_dict(), step=trainer.global_step)
    expected_losses = [trainer.train_step(batch) for _ in range(3)]
    restored = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    restored_optimizer = build_adamw(restored, learning_rate=1e-3)
    restored_scheduler = Scheduler(restored_optimizer, warmup_steps=0, total_steps=6)
    resumed = Trainer(restored, restored_optimizer, scheduler=restored_scheduler)
    state = load_checkpoint(path, restored, optimizer=restored_optimizer, scheduler=restored_scheduler)
    resumed.load_state_dict(state['trainer'])
    assert [resumed.train_step(batch) for _ in range(3)] == pytest.approx(expected_losses)
    assert resumed.global_step == trainer.global_step == 6
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
