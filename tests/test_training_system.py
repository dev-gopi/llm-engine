from pathlib import Path
import math
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader

from datasets.collator import Collator
from datasets.sampler import Sampler
from model.gpt import MiniGPT
from model.loss import CausalLanguageModelLoss
from optim.adamw import build_adamw
from optim.ema import EMA
from optim.scheduler import Scheduler
from training.checkpoint import load_checkpoint, save_checkpoint
from training.distributed import DistributedTrainer
from training.evaluator import Evaluator, aggregate_domain_metrics
from training.trainer import Trainer


def make_loader():
    return DataLoader(
        [torch.tensor([1, 2, 3, 4]), torch.tensor([2, 3, 4])],
        batch_size=2,
        collate_fn=Collator(0),
    )


def test_fit_evaluate_checkpoint_and_resume(tmp_path) -> None:
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    optimizer = build_adamw(model, learning_rate=1e-3)
    scheduler = Scheduler(optimizer, warmup_steps=0, total_steps=2)
    ema = EMA(model, decay=0.9)
    loss_fn = CausalLanguageModelLoss(shift_labels=True)
    trainer = Trainer(model, optimizer, loss_fn, scheduler=scheduler, ema=ema)
    evaluator = Evaluator(model, loss_fn=loss_fn)
    history = trainer.fit(make_loader(), epochs=1, evaluator=evaluator, validation_dataloader=make_loader(), log_every=0)
    assert history[-1]["tokens"] == 5
    assert history[-1]["perplexity"] > 0

    path = save_checkpoint(tmp_path / "model.pt", model, optimizer=optimizer, scheduler=scheduler, ema=ema, step=trainer.global_step)
    restored = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    restored_optimizer = build_adamw(restored)
    restored_scheduler = Scheduler(restored_optimizer, warmup_steps=0, total_steps=2)
    restored_ema = EMA(restored)
    state = load_checkpoint(path, restored, optimizer=restored_optimizer, scheduler=restored_scheduler, ema=restored_ema)
    assert state["step"] == 1
    assert restored_ema.num_updates == 1
    assert restored(torch.tensor([[1, 2]])).shape == (1, 2, 16)


def test_epoch_history_includes_bounded_stability_diagnostics() -> None:
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    trainer = Trainer(model, build_adamw(model, learning_rate=1e-3))
    record = trainer.fit(make_loader(), epochs=1, log_every=0)[-1]
    assert record["loss"] > 0
    assert record["gradient_norm"] > 0
    assert record["parameter_norm"] > 0
    assert record["gradient_to_parameter_ratio"] > 0
    assert record["logit_abs_max"] >= record["logit_abs_mean"]


def test_single_process_distributed_helpers() -> None:
    context = DistributedTrainer.initialize()
    assert context.world_size == 1
    value = torch.tensor(2.0)
    assert DistributedTrainer.mean(value, context).item() == 2.0


def test_checkpoint_can_apply_ema_weights(tmp_path) -> None:
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    ema = EMA(model, decay=0.9)
    expected = ema.shadow["tok.embedding.weight"].clone()
    with torch.no_grad():
        model.tok.weight.add_(2)
    path = save_checkpoint(tmp_path / "ema.pt", model, ema=ema)
    restored = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    state = load_checkpoint(path, restored, use_ema=True)
    assert state["ema_applied"]
    torch.testing.assert_close(restored.tok.weight, expected)


def test_checkpoint_skips_empty_disabled_scaler_state(tmp_path) -> None:
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    disabled_scaler = torch.amp.GradScaler("cuda", enabled=False)
    path = save_checkpoint(tmp_path / "disabled-scaler.pt", model, scaler=disabled_scaler)

    class FreshEnabledScaler:
        def load_state_dict(self, _state):
            raise AssertionError("empty scaler state must not be restored")

    load_checkpoint(path, model, scaler=FreshEnabledScaler())


def test_evaluator_uses_bf16_autocast_and_restores_training_mode() -> None:
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    model.train()
    observed = []
    original_forward = model.forward

    def recording_forward(*args, **kwargs):
        observed.append(torch.is_autocast_enabled("cpu"))
        return original_forward(*args, **kwargs)

    model.forward = recording_forward
    metrics = Evaluator(model, device="cpu", mixed_precision="bf16").evaluate(make_loader())

    assert observed and all(observed)
    assert metrics["tokens"] == 5
    assert model.training


def test_evaluator_does_not_update_training_state() -> None:
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    optimizer = build_adamw(model, learning_rate=1e-3)
    model.train()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]
    gradients_before = [parameter.grad.detach().clone() for parameter in model.parameters()]
    optimizer_before = optimizer.state_dict()
    grad_modes = []
    original_forward = model.forward

    def recording_forward(*args, **kwargs):
        grad_modes.append(torch.is_grad_enabled())
        return original_forward(*args, **kwargs)

    model.forward = recording_forward
    Evaluator(model).evaluate(make_loader())

    assert grad_modes and not any(grad_modes)
    assert model.training
    for parameter, expected_parameter, expected_gradient in zip(
        model.parameters(), parameters_before, gradients_before, strict=True
    ):
        torch.testing.assert_close(parameter, expected_parameter)
        torch.testing.assert_close(parameter.grad, expected_gradient)
    assert optimizer.state_dict() == optimizer_before


def test_evaluator_validates_and_falls_back_from_cpu_fp16() -> None:
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    assert Evaluator(model, device="cpu", mixed_precision="fp16").mixed_precision == "none"
    import pytest
    with pytest.raises(ValueError, match="mixed_precision"):
        Evaluator(model, mixed_precision="fp8")


def test_evaluator_caps_batches_and_logs_progress() -> None:
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    batches = list(make_loader()) * 3

    with patch("training.evaluator.logger.info") as log_info:
        metrics = Evaluator(model).evaluate(
            batches, max_batches=2, progress_every=1, label="chat"
        )

    assert metrics["batches"] == 2
    messages = [call.args[0] % call.args[1:] for call in log_info.call_args_list]
    assert any("validation_progress name=chat batches=1/2" in message for message in messages)
    assert any("validation_progress name=chat batches=2/2" in message for message in messages)


def test_trainer_applies_validation_limit_to_each_domain() -> None:
    class RecordingEvaluator:
        def __init__(self):
            self.calls = []

        def evaluate(self, _loader, **kwargs):
            self.calls.append(kwargs)
            return {"loss": 2.0, "cross_entropy": 2.0, "perplexity": 1.0,
                    "tokens": 1, "batches": 1, "z_loss": 0.0}

    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    evaluator = RecordingEvaluator()
    Trainer(model, build_adamw(model)).fit(
        make_loader(), epochs=1, evaluator=evaluator,
        validation_dataloader={"chat": make_loader(), "math": make_loader()},
        validation_weights={"chat": 0.5, "math": 0.5},
        validation_max_batches=250, validation_progress_every=25,
        log_every=0,
    )

    assert evaluator.calls == [
        {"max_batches": 250, "progress_every": 25, "label": "chat"},
        {"max_batches": 250, "progress_every": 25, "label": "math"},
    ]


def test_domain_validation_uses_explicit_capability_weights() -> None:
    metrics = aggregate_domain_metrics({
        "tinystories": {
            "loss": 1.5, "cross_entropy": 1.4, "z_loss": 1.0,
            "perplexity": 0.0, "tokens": 1000, "batches": 10,
        },
        "wikitext_103": {
            "loss": 3.5, "cross_entropy": 3.4, "z_loss": 3.0,
            "perplexity": 0.0, "tokens": 10, "batches": 2,
        },
    }, {"tinystories": 0.35, "wikitext_103": 0.65})

    assert metrics["loss"] == pytest.approx(2.8)
    assert metrics["cross_entropy"] == pytest.approx(2.7)
    assert metrics["perplexity"] == pytest.approx(math.exp(2.7))
    assert metrics["tokens"] == 1010
    assert metrics["batches"] == 12


def test_early_stopping_tracks_best_validation_epoch() -> None:
    class FixedEvaluator:
        def __init__(self):
            self.losses = iter([2.0, 2.1, 2.2])

        def evaluate(self, _loader):
            return {"loss": next(self.losses)}

    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    optimizer = build_adamw(model, learning_rate=1e-3)
    trainer = Trainer(model, optimizer)
    best_epochs = []
    history = trainer.fit(
        make_loader(), epochs=5, evaluator=FixedEvaluator(),
        validation_dataloader=make_loader(), log_every=0,
        early_stopping_patience=2,
        best_checkpoint_callback=lambda _trainer, epoch: best_epochs.append(epoch),
    )
    assert trainer.stopped_early
    assert len(history) == 3
    assert best_epochs == [0]
    assert trainer.best_validation_loss == 2.0


def test_validation_plateau_reduces_remaining_learning_rate_curve() -> None:
    class FixedEvaluator:
        def __init__(self):
            self.losses = iter([2.0, 2.1, 2.1])

        def evaluate(self, _loader):
            loss = next(self.losses)
            return {"loss": loss, "cross_entropy": loss, "perplexity": 1.0,
                    "tokens": 1, "batches": 1, "z_loss": 0.0}

    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    optimizer = build_adamw(model, learning_rate=1e-3)
    scheduler = Scheduler(
        optimizer, warmup_steps=0, total_steps=2, schedule="constant"
    )
    trainer = Trainer(model, optimizer, scheduler=scheduler)

    trainer.fit(
        list(make_loader()) * 2, epochs=1, evaluator=FixedEvaluator(),
        validation_dataloader=make_loader(), evaluate_every=1, log_every=0,
        validation_lr_decay_factor=0.5, validation_lr_patience=1,
        validation_lr_min_scale=0.25,
    )

    assert trainer.epochs_without_improvement == 1
    assert scheduler.validation_scale == pytest.approx(0.5)
    assert trainer.learning_rate == pytest.approx(5e-4)


def test_validation_lr_decay_respects_minimum_step_spacing() -> None:
    class FixedEvaluator:
        def evaluate(self, _loader):
            return {"loss": 2.0, "cross_entropy": 2.0, "perplexity": 1.0,
                    "tokens": 1, "batches": 1, "z_loss": 0.0}

    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    optimizer = build_adamw(model, learning_rate=1e-3)
    scheduler = Scheduler(optimizer, warmup_steps=0, total_steps=4, schedule="constant")
    trainer = Trainer(model, optimizer, scheduler=scheduler)

    trainer.fit(
        list(make_loader()) * 4, epochs=1, evaluator=FixedEvaluator(),
        validation_dataloader=make_loader(), evaluate_every=1, log_every=0,
        validation_lr_decay_factor=0.5, validation_lr_patience=1,
        validation_lr_min_scale=0.1, validation_lr_min_steps_between_decays=2,
    )

    assert scheduler.validation_scale == pytest.approx(0.25)


def test_generation_accuracy_does_not_block_loss_best_checkpoint() -> None:
    class FixedEvaluator:
        def __init__(self):
            self.losses = iter([2.0, 1.9, 1.9])

        def evaluate(self, _loader):
            loss = next(self.losses)
            return {"loss": loss, "cross_entropy": loss, "perplexity": 1.0,
                    "tokens": 1, "batches": 1, "z_loss": 0.0}

    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    trainer = Trainer(model, build_adamw(model, learning_rate=1e-3))
    saved = []
    accuracies = iter([0.05, 0.25, 0.25])

    trainer.fit(
        list(make_loader()) * 2, epochs=1, evaluator=FixedEvaluator(),
        validation_dataloader=make_loader(), evaluate_every=1, log_every=0,
        validation_callback=lambda *_args: {"accuracy": next(accuracies)},
        best_checkpoint_min_generation_accuracy=0.20,
        best_checkpoint_callback=lambda current, _epoch: saved.append(current.global_step),
    )

    assert saved == [1, 2]
    assert trainer.best_validation_loss == pytest.approx(1.9)


def test_retention_regression_does_not_block_loss_best_checkpoint() -> None:
    class FixedEvaluator:
        def evaluate(self, _loader):
            return {"loss": 1.0, "cross_entropy": 1.0, "perplexity": 1.0,
                    "tokens": 1, "batches": 1, "z_loss": 0.0}

    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    trainer = Trainer(model, build_adamw(model, learning_rate=1e-3))
    saved = []

    trainer.fit(
        make_loader(), epochs=1, evaluator=FixedEvaluator(),
        validation_dataloader=make_loader(), evaluate_every=1, log_every=0,
        validation_callback=lambda *_args: {
            "accuracy": 0.9, "retention_passed": False,
        },
        best_checkpoint_min_generation_accuracy=0.20,
        best_checkpoint_callback=lambda current, _epoch: saved.append(current.global_step),
    )

    assert saved == [1]
    assert trainer.best_validation_loss == 1.0


def test_chat_control_and_domain_gate_reject_hidden_regression() -> None:
    class DomainEvaluator:
        def __init__(self):
            self.losses = iter([2.0, 4.0, 2.2, 1.0, 2.2, 1.0])

        def evaluate(self, _loader):
            loss = next(self.losses)
            return {"loss": loss, "cross_entropy": loss, "perplexity": 1.0,
                    "tokens": 1, "batches": 1, "z_loss": 0.0}

    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    trainer = Trainer(model, build_adamw(model, learning_rate=1e-3))
    saved = []
    trainer.fit(
        list(make_loader()) * 2, epochs=1, evaluator=DomainEvaluator(),
        validation_dataloader={"chat": make_loader(), "math": make_loader()},
        validation_weights={"chat": 0.5, "math": 0.5},
        evaluate_every=1, log_every=0,
        validation_control_domain="chat",
        best_checkpoint_domain_max_regression={"chat": 0.02},
        best_checkpoint_callback=lambda current, _epoch: saved.append(
            current.best_validation_loss
        ),
    )

    assert saved == [3.0]
    assert trainer.best_validation_loss == 3.0
    assert trainer.best_validation_domains["chat"] == 2.0
    assert trainer.epochs_without_improvement == 1


def test_step_zero_validation_is_the_domain_retention_baseline() -> None:
    class DomainEvaluator:
        def __init__(self):
            self.losses = iter([2.0, 3.0, 2.2, 3.2, 2.2, 3.2])

        def evaluate(self, _loader):
            loss = next(self.losses)
            return {"loss": loss, "cross_entropy": loss, "perplexity": 1.0,
                    "tokens": 1, "batches": 1, "z_loss": 0.0}

    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    trainer = Trainer(model, build_adamw(model, learning_rate=1e-3))
    saved = []
    history = trainer.fit(
        make_loader(), epochs=1, evaluator=DomainEvaluator(),
        validation_dataloader={"chat": make_loader(), "knowledge": make_loader()},
        validation_weights={"chat": 0.5, "knowledge": 0.5},
        validation_evaluate_at_start=True, evaluate_every=1, log_every=0,
        best_checkpoint_domain_max_regression={"knowledge": 0.01},
        best_checkpoint_callback=lambda current, _epoch: saved.append(current.global_step),
    )

    assert history[0]["initial_validation"] is True
    assert history[0]["step"] == 0
    assert trainer.best_validation_domains == {"chat": 2.0, "knowledge": 3.0}
    assert saved == []


def test_validation_lr_scale_round_trips_in_scheduler_checkpoint() -> None:
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    optimizer = build_adamw(model, learning_rate=1e-3)
    scheduler = Scheduler(
        optimizer, warmup_steps=0, total_steps=4, schedule="constant"
    )
    scheduler.reduce_after_validation(0.5, min_scale=0.25)

    restored_optimizer = build_adamw(model, learning_rate=1e-3)
    restored = Scheduler(
        restored_optimizer, warmup_steps=0, total_steps=4, schedule="constant"
    )
    restored_optimizer.load_state_dict(optimizer.state_dict())
    restored.load_state_dict(scheduler.state_dict())
    restored_optimizer.step()
    restored.step()

    assert restored.validation_scale == pytest.approx(0.5)
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(5e-4)

    previous, current = restored.reduce_after_validation(0.5, min_scale=0.75)
    assert previous == current == pytest.approx(0.5)


def test_periodic_validation_saves_best_checkpoint_immediately() -> None:
    class FixedEvaluator:
        def __init__(self):
            self.losses = iter([2.0, 2.1])

        def evaluate(self, _loader):
            loss = next(self.losses)
            return {"loss": loss, "cross_entropy": loss, "perplexity": 1.0,
                    "tokens": 1, "batches": 1, "z_loss": 0.0}

    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    optimizer = build_adamw(model, learning_rate=1e-3)
    trainer = Trainer(model, optimizer)
    saved = []

    trainer.fit(
        make_loader(), epochs=1, evaluator=FixedEvaluator(),
        validation_dataloader=make_loader(), log_every=0, evaluate_every=1,
        best_checkpoint_callback=lambda current, epoch: saved.append(
            (current.global_step, epoch, current.best_validation_loss)
        ),
    )

    assert saved == [(1, 0, 2.0)]
    assert trainer.best_validation_loss == 2.0


def test_initial_validation_can_save_a_guaranteed_baseline_checkpoint() -> None:
    class FixedEvaluator:
        def evaluate(self, _loader):
            return {"loss": 2.5, "cross_entropy": 2.5, "perplexity": 1.0,
                    "tokens": 1, "batches": 1, "z_loss": 0.0}

    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    trainer = Trainer(model, build_adamw(model, learning_rate=1e-3))
    saved = []

    history = trainer.fit(
        make_loader(), epochs=1, evaluator=FixedEvaluator(),
        validation_dataloader=make_loader(), validation_evaluate_at_start=True,
        save_initial_best_checkpoint=True, log_every=0,
        best_checkpoint_min_generation_accuracy=1.0,
        best_checkpoint_callback=lambda current, epoch: saved.append(
            (current.global_step, epoch, current.best_validation_loss)
        ),
    )

    assert history[0]["initial_validation"] is True
    assert saved == [(0, -1, 2.5)]
    assert trainer.best_validation_loss == 2.5


def test_initial_best_repairs_resumed_state_with_infinite_best_loss() -> None:
    class FixedDomainEvaluator:
        losses = iter([2.0, 3.0, 2.1, 3.1])

        def evaluate(self, _loader):
            loss = next(self.losses)
            return {"loss": loss, "cross_entropy": loss, "perplexity": 1.0,
                    "tokens": 1, "batches": 1, "z_loss": 0.0}

    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    trainer = Trainer(model, build_adamw(model, learning_rate=1e-3))
    trainer.global_step = 5000
    trainer.best_validation_domains = {"chat": 9.0, "knowledge": 9.0}
    saved = []

    trainer.fit(
        make_loader(), epochs=1, evaluator=FixedDomainEvaluator(),
        validation_dataloader={"chat": make_loader(), "knowledge": make_loader()},
        validation_weights={"chat": 0.5, "knowledge": 0.5},
        validation_evaluate_at_start=True, save_initial_best_checkpoint=True,
        log_every=0,
        best_checkpoint_callback=lambda current, _epoch: saved.append(
            current.best_validation_loss
        ),
    )

    assert saved == [2.5]
    assert trainer.best_validation_domains == {"chat": 2.0, "knowledge": 3.0}


def test_validation_callback_runs_after_periodic_validation() -> None:
    class FixedEvaluator:
        def evaluate(self, _loader):
            return {"loss": 2.0, "cross_entropy": 2.0, "perplexity": 1.0,
                    "tokens": 1, "batches": 1, "z_loss": 0.0}

    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    trainer = Trainer(model, build_adamw(model, learning_rate=1e-3))
    observed = []

    history = trainer.fit(
        make_loader(), epochs=1, evaluator=FixedEvaluator(),
        validation_dataloader=make_loader(), log_every=0, evaluate_every=1,
        validation_callback=lambda current, epoch, metrics, domains: observed.append(
            (current.global_step, epoch, metrics["loss"], domains)
        ) or {"accuracy": 0.5},
    )

    assert observed[0] == (1, 0, 2.0, {})
    assert history[0]["generation_evaluation"] == {"accuracy": 0.5}


def test_periodic_latest_checkpoint_contains_same_step_validation_state() -> None:
    class FixedEvaluator:
        def __init__(self):
            self.losses = iter([2.0, 2.1])

        def evaluate(self, _loader):
            loss = next(self.losses)
            return {"loss": loss, "cross_entropy": loss, "perplexity": 1.0,
                    "tokens": 1, "batches": 1, "z_loss": 0.0}

    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    optimizer = build_adamw(model, learning_rate=1e-3)
    trainer = Trainer(model, optimizer)
    latest_states = []

    trainer.fit(
        make_loader(), epochs=1, evaluator=FixedEvaluator(),
        validation_dataloader=make_loader(), log_every=0,
        evaluate_every=1, checkpoint_every=1,
        checkpoint_callback=lambda current, _epoch: latest_states.append(
            current.state_dict()
        ),
        validation_metric_name="dataset_weighted_v1",
    )

    assert latest_states[0]["global_step"] == 1
    assert math.isinf(latest_states[0]["best_validation_loss"])
    assert latest_states[1]["best_validation_loss"] == 2.0
    assert latest_states[1]["early_stopping_best_loss"] == 2.0
    assert latest_states[1]["validation_metric_name"] == "dataset_weighted_v1"


def test_best_checkpoint_keeps_small_improvement_below_early_stopping_delta() -> None:
    class FixedEvaluator:
        def __init__(self):
            self.losses = iter([2.0, 1.9995])

        def evaluate(self, _loader):
            loss = next(self.losses)
            return {"loss": loss, "cross_entropy": loss, "perplexity": 1.0,
                    "tokens": 1, "batches": 1, "z_loss": 0.0}

    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    optimizer = build_adamw(model, learning_rate=1e-3)
    trainer = Trainer(model, optimizer)
    saved_losses = []

    trainer.fit(
        make_loader(), epochs=1, evaluator=FixedEvaluator(),
        validation_dataloader=make_loader(), log_every=0, evaluate_every=1,
        early_stopping_min_delta=0.001,
        best_checkpoint_callback=lambda current, _epoch: saved_losses.append(
            current.best_validation_loss
        ),
    )

    assert saved_losses == [2.0, 1.9995]
    assert trainer.best_validation_loss == 1.9995
    assert trainer.early_stopping_best_loss == 2.0


def test_changed_validation_metric_resets_incompatible_best_baseline() -> None:
    class FixedEvaluator:
        def evaluate(self, _loader):
            return {"loss": 2.8, "cross_entropy": 2.7, "perplexity": 1.0,
                    "tokens": 1, "batches": 1, "z_loss": 0.0}

    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    optimizer = build_adamw(model, learning_rate=1e-3)
    trainer = Trainer(model, optimizer)
    trainer.best_validation_loss = 1.6
    trainer.early_stopping_best_loss = 1.6
    saved = []

    trainer.fit(
        make_loader(), epochs=1, evaluator=FixedEvaluator(),
        validation_dataloader=make_loader(), log_every=0,
        validation_metric_name="dataset_weighted_v1",
        best_checkpoint_callback=lambda current, _epoch: saved.append(
            current.best_validation_loss
        ),
    )

    assert saved == [2.8]
    assert trainer.best_validation_loss == 2.8
    assert trainer.validation_metric_name == "dataset_weighted_v1"


def test_resumed_epoch_loss_uses_batches_processed_after_resume() -> None:
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    optimizer = build_adamw(model, learning_rate=1e-3)
    trainer = Trainer(model, optimizer)
    trainer.batch_in_epoch = 5

    losses = iter([2.0, 4.0])

    def train_step(_batch):
        trainer.global_step += 1
        return next(losses)

    trainer.train_step = train_step
    with patch("training.trainer.logger.info") as log_info:
        history = trainer.fit([{}, {}], epochs=1, log_every=1)

    assert history[-1]["train_loss"] == 3.0
    assert [call.args[-1] for call in log_info.call_args_list] == [2.0, 3.0]


def test_resumed_progress_uses_full_epoch_length() -> None:
    class ResumeLoader:
        def __init__(self):
            self.batch_sampler = Sampler(list(range(200)), 2, shuffle=False)
            self.batch_sampler.set_start_batch(50)

        def __len__(self):
            return len(self.batch_sampler)

        def __iter__(self):
            yield {}

    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    trainer = Trainer(model, build_adamw(model, learning_rate=1e-3))
    trainer.current_epoch = 2
    trainer.batch_in_epoch = 50

    def train_step(_batch):
        trainer.global_step += 1
        return 2.0

    trainer.train_step = train_step
    with patch("training.trainer.logger.info") as log_info:
        trainer.fit(ResumeLoader(), epochs=3, log_every=1)

    progress_percent = log_info.call_args_list[0].args[9]
    epoch_progress_percent = log_info.call_args_list[0].args[10]
    assert progress_percent == pytest.approx(251 / 300 * 100)
    assert epoch_progress_percent == pytest.approx(51)
    assert progress_percent < 100


def test_checkpoint_architecture_mismatch_error(tmp_path) -> None:
    import pytest
    model_small = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    path = save_checkpoint(tmp_path / "small.pt", model_small, metadata={"model_config": {"dim": 8, "layers": 1}})

    model_large = MiniGPT(vocab_size=16, dim=16, layers=2, heads=2, max_pos=8)
    with pytest.raises(RuntimeError, match="architecture mismatch"):
        load_checkpoint(path, model_large)


def test_low_memory_checkpoint_load_preserves_weights_and_tying(tmp_path) -> None:
    source = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    path = save_checkpoint(tmp_path / "low-memory.pt", source)
    restored = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    load_checkpoint(path, restored, low_memory=True, restore_rng=False)
    assert restored.head.weight is restored.tok.weight
    for name, expected in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected)


def test_low_memory_checkpoint_requires_cpu_mapping(tmp_path) -> None:
    import pytest
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    path = save_checkpoint(tmp_path / "low-memory.pt", model)
    with pytest.raises(ValueError, match="map_location='cpu'"):
        load_checkpoint(path, model, low_memory=True, map_location="meta")


def test_checkpoint_rejects_different_same_size_tokenizer(tmp_path) -> None:
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    path = save_checkpoint(
        tmp_path / "model.pt", model,
        metadata={"tokenizer_fingerprint": "tokenizer-a"},
    )

    import pytest
    with pytest.raises(ValueError, match="tokenizer fingerprint"):
        load_checkpoint(
            path, model, expected_tokenizer_fingerprint="tokenizer-b"
        )


def test_checkpoint_loads_verified_append_only_vocabulary_extension(tmp_path) -> None:
    original = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    path = save_checkpoint(
        tmp_path / "model.pt", original,
        metadata={"tokenizer_fingerprint": "base-tokenizer"},
    )
    extended = MiniGPT(vocab_size=19, dim=8, layers=1, heads=2, max_pos=8)
    expected_new_rows = original.tok.weight.mean(dim=0).expand(3, -1)

    load_checkpoint(
        path,
        extended,
        expected_tokenizer_fingerprint="extended-tokenizer",
        compatible_tokenizer_fingerprints={"base-tokenizer"},
        allow_vocab_extension=True,
        restore_rng=False,
    )

    torch.testing.assert_close(extended.tok.weight[:16], original.tok.weight)
    torch.testing.assert_close(extended.tok.weight[16:], expected_new_rows)
    assert extended.head.weight is extended.tok.weight


def test_checkpoint_expands_ema_for_append_only_vocabulary(tmp_path) -> None:
    original = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    ema = EMA(original, decay=0.9)
    expected_prefix = ema.shadow["tok.embedding.weight"].clone()
    path = save_checkpoint(
        tmp_path / "model.pt", original, ema=ema,
        metadata={"tokenizer_fingerprint": "base-tokenizer"},
    )
    extended = MiniGPT(vocab_size=19, dim=8, layers=1, heads=2, max_pos=8)
    expected_new_rows = expected_prefix.mean(dim=0).expand(3, -1)

    load_checkpoint(
        path,
        extended,
        use_ema=True,
        expected_tokenizer_fingerprint="extended-tokenizer",
        compatible_tokenizer_fingerprints={"base-tokenizer"},
        allow_vocab_extension=True,
        restore_rng=False,
    )

    torch.testing.assert_close(extended.tok.weight[:16], expected_prefix)
    torch.testing.assert_close(extended.tok.weight[16:], expected_new_rows)


def test_checkpoint_rng_state_loading(tmp_path) -> None:
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    path = save_checkpoint(tmp_path / "rng.pt", model)
    restored = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)

    # With restore_rng=False
    load_checkpoint(path, restored, restore_rng=False)

    # With restore_rng=True
    load_checkpoint(path, restored, restore_rng=True)


def test_evaluation_sum_and_mean_reductions_agree():
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    batches = [
        {"input_ids": torch.tensor([[1, 2, 3, 4]]),
         "labels": torch.tensor([[1, 2, 3, 4]])},
        {"input_ids": torch.tensor([[2, 3, 4]]),
         "labels": torch.tensor([[2, -100, 4]])},
    ]
    results = [Evaluator(model, loss_fn=CausalLanguageModelLoss(
        reduction=reduction, z_loss_coefficient=0.01,
    )).evaluate(batches) for reduction in ("mean", "sum")]
    assert results[0] == pytest.approx(results[1])


def test_train_step_reuses_loss_token_count_with_mask(monkeypatch):
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    trainer = Trainer(model, build_adamw(model), CausalLanguageModelLoss())

    def unexpected_recount(*args, **kwargs):
        raise AssertionError("standard loss already counted the valid targets")

    monkeypatch.setattr(trainer, "_count_target_tokens", unexpected_recount)
    value = trainer.train_step({
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[1, 2, -100, 4]]),
        "loss_mask": torch.tensor([[1, 1, 1, 0]]),
    })
    assert math.isfinite(value)
    assert trainer.tokens_processed == 1


def test_custom_loss_keeps_tensor_contract_and_token_count():
    class CustomLoss(torch.nn.Module):
        shift_labels = False
        ignore_index = -100

        def forward(self, logits, labels, *, loss_mask=None):
            return torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten())

    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    trainer = Trainer(model, build_adamw(model), CustomLoss())
    assert math.isfinite(trainer.train_step(torch.tensor([[1, 2]]), torch.tensor([[2, 3]])))
    assert trainer.tokens_processed == 2


def test_evaluator_reduces_metrics_before_host_conversion(monkeypatch):
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    evaluator = Evaluator(model)
    batches = list(make_loader())
    expected = evaluator.evaluate(batches)
    monkeypatch.setattr("training.evaluator.dist.is_initialized", lambda: True)

    def simulated_two_ranks(totals, op):
        assert totals.shape == (5,)
        totals.mul_(2)

    monkeypatch.setattr("training.evaluator.dist.all_reduce", simulated_two_ranks)
    actual = evaluator.evaluate(batches)
    assert actual["loss"] == pytest.approx(expected["loss"])
    assert actual["tokens"] == 2 * expected["tokens"]
    assert actual["batches"] == 2 * expected["batches"]


def test_log_timing_estimates_use_current_window_with_accumulation() -> None:
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    trainer = Trainer(model, build_adamw(model, learning_rate=1e-3),
                      gradient_accumulation_steps=2)
    clock = [0.0]

    def train_step(_batch):
        clock[0] += 2.0
        trainer.micro_step += 1
        if trainer.micro_step % 2 == 0:
            trainer.global_step += 1
        return 1.0

    trainer.train_step = train_step
    with patch("training.trainer.time.perf_counter", side_effect=lambda: clock[0]), \
            patch("training.trainer.logger.info") as log_info:
        trainer.fit([{}] * 8, epochs=1, log_every=1, checkpoint_every=2,
                    checkpoint_callback=lambda *_: None)
    messages = [call.args[0] % call.args[1:] for call in log_info.call_args_list]
    assert "log_interval_seconds=4.00" in messages[0]
    assert "seconds_per_step=4.000" in messages[0]
    assert "next_log_eta_seconds=4.0" in messages[0]
    assert "next_checkpoint_eta_seconds=4.0" in messages[0]
    assert "next_validation_eta_seconds=disabled" in messages[0]
    assert "next_checkpoint_eta_seconds=0.0" in messages[1]
    assert any("checkpoint kind=latest step=2 duration_seconds=0.00" in m for m in messages)


def test_log_interval_seconds_uses_wall_clock_cadence() -> None:
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    trainer = Trainer(model, build_adamw(model, learning_rate=1e-3))
    clock = [0.0]

    def train_step(_batch):
        clock[0] += 60.0
        trainer.micro_step += 1
        trainer.global_step += 1
        return 1.0

    trainer.train_step = train_step
    with patch("training.trainer.time.perf_counter", side_effect=lambda: clock[0]), \
            patch("training.trainer.logger.info") as log_info:
        trainer.fit([{}] * 6, epochs=1, log_every=1, log_interval_seconds=300)

    messages = [call.args[0] % call.args[1:] for call in log_info.call_args_list]
    training_messages = [message for message in messages if message.startswith("epoch=")]
    assert len(training_messages) == 1
    assert "step=5" in training_messages[0]
    assert "log_interval_seconds=300.00" in training_messages[0]


@pytest.mark.parametrize('patience,losses,stop_step', [
    (1, [2.0, 2.1], 2),
    (2, [2.0, 2.1, 1.8, 1.9, 2.0], 5),
])
def test_periodic_early_stopping_preserves_resume_position(patience, losses, stop_step):
    class FixedEvaluator:
        def __init__(self):
            self.losses = iter(losses)

        def evaluate(self, _loader):
            return {"loss": next(self.losses)}

    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    trainer = Trainer(model, build_adamw(model, learning_rate=1e-3))
    saved, best = [], []
    history = trainer.fit(
        list(make_loader()) * 8, epochs=3, evaluator=FixedEvaluator(),
        validation_dataloader=make_loader(), evaluate_every=1, log_every=0,
        early_stopping_patience=patience,
        checkpoint_callback=lambda current, _epoch: saved.append(current.state_dict()),
        best_checkpoint_callback=lambda current, _epoch: best.append(current.best_validation_loss),
    )
    assert trainer.stopped_early
    assert trainer.global_step == stop_step
    assert trainer.current_epoch == 0
    assert trainer.batch_in_epoch == stop_step
    assert len(history) == stop_step
    assert saved[-1]['batch_in_epoch'] == stop_step
    assert saved[-1]['epochs_without_improvement'] == patience
    assert best[-1] == min(losses)


def test_evaluator_selects_ema_and_restores_training_weights():
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    ema = EMA(model, decay=0.9)
    expected = Evaluator(model).evaluate(make_loader())["cross_entropy"]
    with torch.no_grad():
        model.tok.weight.mul_(3)
    trained = model.tok.weight.detach().clone()
    actual = Evaluator(model, ema=ema).evaluate(make_loader())["cross_entropy"]
    assert actual == pytest.approx(expected)
    torch.testing.assert_close(model.tok.weight, trained, rtol=0, atol=0)


def test_reasoning_training_policy_requires_explicit_loss_mask() -> None:
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    trainer = Trainer(model, build_adamw(model), reasoning_trace_policy="assistant_only")
    with pytest.raises(ValueError, match="requires an explicit loss_mask"):
        trainer.train_step({"input_ids": torch.tensor([[1, 2, 3]]), "labels": torch.tensor([[2, 3, 4]])})


def test_reasoning_training_policy_accepts_masked_sft_batch() -> None:
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    trainer = Trainer(model, build_adamw(model, learning_rate=1e-3), reasoning_trace_policy="assistant_only")
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[-100, -100, 3, 4]]),
        "loss_mask": torch.tensor([[False, False, True, True]]),
    }
    assert math.isfinite(trainer.train_step(batch))


def test_reasoning_sft_profile_is_reproducible_and_covers_required_domains() -> None:
    import yaml
    config = yaml.safe_load(Path("configs/finetuning.reasoning.gpu.yaml").read_text())
    profile = config["reasoning_sft"]
    assert profile["schema_version"] == 1
    assert profile["trace_open"] == "<thinking>"
    assert profile["trace_close"] == "</thinking>"
    assert profile["allow_untraced_assistant_examples"] is False
    assert set(profile["domains"]) == {"math", "code", "logic", "planning", "verification", "self_correction"}
    assert math.isclose(sum(profile["weights"].values()), 1.0)
    assert config["reasoning_trace_policy"] == "assistant_only"
    assert config["seed"] == 20260921
    for path in [*config["train_files"], *config["validation_files"]]:
        assert Path(path).is_file()
        assert (Path(path).parent / "dataset-manifest.yaml").is_file()
