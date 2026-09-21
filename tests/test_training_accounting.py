import pytest
import torch

from training.accounting import TrainingAccounting
from training.trainer import Trainer


def test_training_accounting_is_deterministic_and_reports_units() -> None:
    accounting = TrainingAccounting(100, 1_000, 10, 20, device_count=2)
    report = accounting.report()
    assert report["estimated_flops"] == 600_000
    assert report["device_hours"] == pytest.approx(1 / 90)
    assert report["tokens_per_second"] == 50
    assert report["flop_estimate_multiplier"] == 6


@pytest.mark.parametrize("kwargs", [
    {"parameters": 0, "supervised_tokens": 1, "optimizer_steps": 1, "elapsed_seconds": 1},
    {"parameters": 1, "supervised_tokens": -1, "optimizer_steps": 1, "elapsed_seconds": 1},
    {"parameters": 1, "supervised_tokens": 1, "optimizer_steps": 1, "elapsed_seconds": -1},
])
def test_training_accounting_rejects_invalid_counters(kwargs) -> None:
    with pytest.raises(ValueError):
        TrainingAccounting(**kwargs)


def test_trainer_accounting_uses_persisted_counters() -> None:
    model = torch.nn.Linear(2, 2)
    trainer = Trainer(model, torch.optim.AdamW(model.parameters()))
    trainer.tokens_processed, trainer.global_step, trainer.training_seconds = 50, 3, 10
    report = trainer.accounting_report()
    assert report["parameters"] == sum(parameter.numel() for parameter in model.parameters())
    assert report["supervised_tokens"] == 50
    assert report["optimizer_steps"] == 3
