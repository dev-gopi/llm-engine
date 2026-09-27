import copy
import json
import subprocess
import sys
from pathlib import Path

import torch

from model.gpt import MiniGPT
from optim.adamw import build_adamw
from optim.scheduler import Scheduler
from post_training.dpo import DPOTrainer
from post_training.preference_data import build_preference_loader
from scripts.prepare_helpsteer_preferences import quality
from tokenizer.bpe import BYTE_ENCODER
from tokenizer.encoder import DEFAULT_SPECIAL_TOKENS, Tokenizer
from training.checkpoint import load_checkpoint, save_checkpoint


def test_helpsteer_quality_aggregates_all_dimensions() -> None:
    record = {"scores": {
        "helpfulness": 4, "correctness": 0, "coherence": 2,
        "complexity": 1, "verbosity": 3,
    }}
    assert quality(record) == 2.0


def tokenizer() -> Tokenizer:
    pieces = list(DEFAULT_SPECIAL_TOKENS) + list(BYTE_ENCODER.values())
    vocab = {piece: index for index, piece in enumerate(pieces)}
    return Tokenizer(vocab, special_tokens={piece: vocab[piece] for piece in DEFAULT_SPECIAL_TOKENS})


def preference_file(path: Path) -> Path:
    records = [
        {"prompt": "Say hello", "chosen": "Hello!", "rejected": "Go away."},
        {"prompt": "What is two plus two?", "chosen": "Four.", "rejected": "Five."},
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    return path


def test_dpo_fit_evaluate_best_checkpoint_and_resume(tmp_path) -> None:
    tok = tokenizer()
    source = preference_file(tmp_path / "preferences.jsonl")
    train_loader = build_preference_loader(
        [str(source)], tok, max_length=64, batch_size=1, shuffle=False,
    )
    policy = MiniGPT(vocab_size=tok.vocab_size, dim=8, layers=1, heads=2, max_pos=64)
    reference = copy.deepcopy(policy)
    optimizer = build_adamw(policy, learning_rate=1e-4)
    scheduler = Scheduler(optimizer, warmup_steps=0, total_steps=2)
    trainer = DPOTrainer(policy, reference, optimizer, scheduler=scheduler)
    best_path = tmp_path / "best.pt"

    history = trainer.fit(
        train_loader, epochs=1, validation_loader=train_loader, log_every=0,
        best_checkpoint_callback=lambda current, _epoch: save_checkpoint(
            best_path, policy, optimizer=optimizer, scheduler=scheduler,
            scaler=current.scaler, step=current.global_step, trainer=current.state_dict(),
        ),
    )

    assert trainer.global_step == 2
    assert history[0]["validation_loss"] > 0
    assert 0 <= history[0]["validation_reward_accuracy"] <= 1
    restored = MiniGPT(vocab_size=tok.vocab_size, dim=8, layers=1, heads=2, max_pos=64)
    restored_optimizer = build_adamw(restored, learning_rate=1e-4)
    restored_scheduler = Scheduler(restored_optimizer, warmup_steps=0, total_steps=2)
    state = load_checkpoint(
        best_path, restored, optimizer=restored_optimizer, scheduler=restored_scheduler,
    )
    assert state["step"] == 2
    assert state["trainer"]["current_epoch"] == 1


def test_dpo_cli_and_profiles_are_available() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "scripts/train_dpo.py", "--help"], cwd=root,
        text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--reference-checkpoint" in completed.stdout
    assert (root / "configs/dpo.cpu.yaml").is_file()
    assert (root / "configs/dpo.gpu.yaml").is_file()


def test_preferences_reject_missing_fields_and_truncated_answers() -> None:
    from post_training.preference_data import PreferenceDataset

    records = [
        {"prompt": "Q", "chosen": "Yes", "rejected": "No"},
        {"prompt": None, "chosen": "Yes", "rejected": "No"},
        {"prompt": "Q", "chosen": None, "rejected": "No"},
        {"prompt": "Q" * 100, "chosen": "Yes", "rejected": "No"},
        {"prompt": "Q", "chosen": "A" * 100, "rejected": "No"},
    ]
    dataset = PreferenceDataset(records, tokenizer(), max_length=32)
    assert len(dataset) == 1
    assert dataset[0]["chosen_mask"].any()
    assert dataset[0]["rejected_mask"].any()


def test_dpo_evaluation_weights_pairs_not_batches(tmp_path) -> None:
    tok = tokenizer()
    source = preference_file(tmp_path / "preferences.jsonl")
    with source.open("a") as stream:
        stream.write(json.dumps({"prompt": "Hi", "chosen": "Hello", "rejected": "Bye"}) + "\n")
    policy = MiniGPT(vocab_size=tok.vocab_size, dim=8, layers=1, heads=2, max_pos=64)
    trainer = DPOTrainer(policy, copy.deepcopy(policy), build_adamw(policy, learning_rate=1e-4))

    def batch_loss(values):
        score = values["chosen_attention_mask"].sum(dim=1).float().mean()
        return score, {"reward_accuracy": score / 100, "reward_margin": score}

    trainer._batch_loss = batch_loss
    results = [trainer.evaluate(build_preference_loader(
        [str(source)], tok, max_length=64, batch_size=size, shuffle=False,
    )) for size in (1, 2, 3)]
    import pytest
    for result in results[1:]:
        assert result == pytest.approx(results[0])


def test_dpo_nonfinite_gradients_without_clipping_do_not_update() -> None:
    import pytest

    policy = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.1)
    trainer = DPOTrainer(policy, copy.deepcopy(policy), optimizer, gradient_clip_norm=None)
    before = {key: value.clone() for key, value in policy.state_dict().items()}
    policy.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    trainer._batch_loss = lambda values: (policy.weight.sum(), {})
    with pytest.raises(FloatingPointError, match="gradients"):
        trainer.train_step({})
    assert trainer.global_step == 0
    for key, value in policy.state_dict().items():
        assert torch.equal(value, before[key])


def test_ipo_loss_and_trainer_method_are_available() -> None:
    from post_training.dpo import IPOLoss

    loss_fn = IPOLoss(beta=0.1)
    policy_chosen = torch.tensor([2.0, 3.0])
    policy_rejected = torch.tensor([1.0, 1.5])
    reference_chosen = torch.tensor([1.0, 1.0])
    reference_rejected = torch.tensor([0.5, 0.5])
    loss, metrics = loss_fn(
        policy_chosen, policy_rejected, reference_chosen, reference_rejected
    )
    assert torch.isfinite(loss)
    assert 0.0 <= float(metrics["reward_accuracy"]) <= 1.0

    policy = torch.nn.Linear(1, 1)
    trainer = DPOTrainer(
        policy, copy.deepcopy(policy), torch.optim.AdamW(policy.parameters(), lr=1e-4),
        method="ipo",
    )
    assert trainer.method == "ipo"
    assert trainer.state_dict()["method"] == "ipo"


def test_preference_loader_supports_distributed_sharding(tmp_path) -> None:
    tok = tokenizer()
    source = tmp_path / "preferences.jsonl"
    rows = [
        {"prompt": f"Q{i}", "chosen": f"good{i}", "rejected": f"bad{i}"}
        for i in range(8)
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    loaders = [
        build_preference_loader(
            [str(source)], tok, max_length=64, batch_size=1, shuffle=True,
            rank=rank, world_size=2, seed=123,
        )
        for rank in (0, 1)
    ]
    assert len(loaders[0]) == len(loaders[1]) == 4
    assert loaders[0].sampler is not None
    assert loaders[1].sampler is not None
    assert set(iter(loaders[0].sampler)).isdisjoint(set(iter(loaders[1].sampler)))


def test_sft_cli_wrapper_and_ipo_profiles_are_available() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "scripts/train_sft.py", "--help"], cwd=root,
        text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--training-config" in completed.stdout
    assert (root / "configs/ipo.cpu.yaml").is_file()
    assert (root / "configs/ipo.gpu.yaml").is_file()


def test_orpo_loss_and_reference_free_trainer_are_available() -> None:
    from post_training.dpo import ORPOLoss

    loss_fn = ORPOLoss(preference_weight=0.1)
    chosen = torch.tensor([-0.2, -0.4])
    rejected = torch.tensor([-1.0, -0.8])
    loss, metrics = loss_fn(chosen, rejected)
    assert torch.isfinite(loss)
    assert float(metrics["reward_accuracy"]) == 1.0
    assert float(metrics["reward_margin"]) > 0

    policy = torch.nn.Linear(1, 1)
    trainer = DPOTrainer(
        policy, None, torch.optim.AdamW(policy.parameters(), lr=1e-4),
        method="orpo", orpo_lambda=0.1,
    )
    assert trainer.reference is None
    assert trainer.method == "orpo"


def test_orpo_profiles_are_available() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "configs/orpo.cpu.yaml").is_file()
    assert (root / "configs/orpo.gpu.yaml").is_file()


def test_kto_loss_binary_feedback_and_trainer_method() -> None:
    from post_training.dpo import KTOLoss

    loss_fn = KTOLoss(beta=0.1)
    policy = torch.tensor([-1.0, -2.0, -1.5, -3.0])
    reference = torch.tensor([-1.5, -1.5, -1.7, -2.0])
    desirable = torch.tensor([True, False, True, False])
    loss, metrics = loss_fn(policy, reference, desirable)
    assert torch.isfinite(loss)
    assert 0.0 <= float(metrics["reward_accuracy"]) <= 1.0

    model = torch.nn.Linear(1, 1)
    trainer = DPOTrainer(
        model, copy.deepcopy(model), torch.optim.AdamW(model.parameters(), lr=1e-4),
        method="kto",
    )
    assert trainer.method == "kto"
    assert trainer.state_dict()["method"] == "kto"


def test_kto_loader_accepts_native_labels_and_pair_migration(tmp_path) -> None:
    from post_training.preference_data import build_kto_loader

    tok = tokenizer()
    source = tmp_path / "kto.jsonl"
    rows = [
        {"prompt": "Q1", "completion": "good", "label": "desirable"},
        {"prompt": "Q2", "completion": "bad", "label": "undesirable"},
        {"prompt": "Q3", "chosen": "yes", "rejected": "no"},
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    loader = build_kto_loader([str(source)], tok, max_length=64, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    assert batch["completion_ids"].shape[0] == 4
    assert batch["desirable"].tolist().count(True) == 2
    assert batch["desirable"].tolist().count(False) == 2
    assert batch["completion_mask"].any()


def test_kto_profiles_and_cli_are_available() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "scripts/train_dpo.py", "--help"], cwd=root,
        text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "kto" in completed.stdout
    assert (root / "configs/kto.cpu.yaml").is_file()
    assert (root / "configs/kto.gpu.yaml").is_file()


def test_kto_loss_accepts_single_class_minibatches() -> None:
    from post_training.dpo import KTOLoss
    loss_fn = KTOLoss(beta=0.1)
    for labels in (torch.tensor([True, True]), torch.tensor([False, False])):
        loss, metrics = loss_fn(torch.tensor([-1.0, -1.2]), torch.tensor([-1.1, -1.1]), labels)
        assert torch.isfinite(loss)
        assert 0.0 <= float(metrics["reward_accuracy"]) <= 1.0
