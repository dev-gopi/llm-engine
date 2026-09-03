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


def test_evaluator_validates_and_falls_back_from_cpu_fp16() -> None:
    model = MiniGPT(vocab_size=16, dim=8, layers=1, heads=2, max_pos=8)
    assert Evaluator(model, device="cpu", mixed_precision="fp16").mixed_precision == "none"
    import pytest
    with pytest.raises(ValueError, match="mixed_precision"):
        Evaluator(model, mixed_precision="fp8")


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

    assert latest_states[0]["best_validation_loss"] == 2.0
    assert latest_states[0]["early_stopping_best_loss"] == 2.0
    assert latest_states[0]["validation_metric_name"] == "dataset_weighted_v1"


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

    progress_percent = log_info.call_args_list[0].args[8]
    epoch_progress_percent = log_info.call_args_list[0].args[9]
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
