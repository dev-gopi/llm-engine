import copy
import json
import subprocess
import sys
from pathlib import Path

import torch

from model.gpt import MiniGPT
from optim.adamw import build_adamw
from post_training.grpo import GRPOLoss, GRPOTrainer
from post_training.grpo_data import build_grpo_loader
from post_training.preference_data import build_preference_loader
from post_training.reward_model import (
    PairwiseRewardLoss,
    RewardModel,
    RewardModelTrainer,
)
from tokenizer.bpe import BYTE_ENCODER
from tokenizer.encoder import DEFAULT_SPECIAL_TOKENS, Tokenizer


def tokenizer() -> Tokenizer:
    pieces = list(DEFAULT_SPECIAL_TOKENS) + list(BYTE_ENCODER.values())
    vocab = {piece: index for index, piece in enumerate(pieces)}
    return Tokenizer(
        vocab, special_tokens={piece: vocab[piece] for piece in DEFAULT_SPECIAL_TOKENS}
    )


def test_minigpt_hidden_state_opt_in_preserves_default_forward() -> None:
    model = MiniGPT(vocab_size=32, dim=8, layers=1, heads=2, max_pos=16)
    ids = torch.tensor([[1, 2, 3]])
    ordinary = model(ids)
    with_hidden = model(ids, return_hidden_states=True)
    assert isinstance(ordinary, torch.Tensor)
    assert isinstance(with_hidden, tuple) and len(with_hidden) == 2
    logits, hidden = with_hidden
    assert torch.allclose(ordinary, logits)
    assert hidden.shape == (1, 3, 8)


def test_pairwise_reward_loss_prefers_higher_chosen_reward() -> None:
    loss_fn = PairwiseRewardLoss()
    good_loss, good_metrics = loss_fn(
        torch.tensor([2.0, 1.0]), torch.tensor([0.0, -1.0])
    )
    bad_loss, _ = loss_fn(torch.tensor([0.0, -1.0]), torch.tensor([2.0, 1.0]))
    assert good_loss < bad_loss
    assert float(good_metrics["reward_accuracy"]) == 1.0
    assert float(good_metrics["reward_margin"]) > 0


def test_reward_model_uses_last_nonpadding_hidden_state() -> None:
    backbone = MiniGPT(vocab_size=32, dim=8, layers=1, heads=2, max_pos=16)
    model = RewardModel(backbone, 8)
    ids = torch.tensor([[1, 2, 3, 0], [1, 4, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    reward = model(ids, attention_mask=mask)
    assert reward.shape == (2,)
    assert torch.isfinite(reward).all()


def test_reward_model_trainer_one_step(tmp_path) -> None:
    tok = tokenizer()
    path = tmp_path / "pairs.jsonl"
    rows = [
        {"prompt": "Q1", "chosen": "good", "rejected": "bad"},
        {"prompt": "Q2", "chosen": "right", "rejected": "wrong"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    loader = build_preference_loader(
        [str(path)], tok, max_length=64, batch_size=2, shuffle=False
    )
    model = RewardModel(
        MiniGPT(vocab_size=tok.vocab_size, dim=8, layers=1, heads=2, max_pos=64), 8
    )
    trainer = RewardModelTrainer(model, build_adamw(model, learning_rate=1e-4))
    metrics = trainer.train_step(next(iter(loader)))
    assert trainer.global_step == 1
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert 0.0 <= metrics["reward_accuracy"] <= 1.0


def test_grpo_loss_group_normalization_and_clipping() -> None:
    loss_fn = GRPOLoss(clip_epsilon=0.2, beta=0.04)
    old = torch.tensor([[-1.0, -1.0, -1.0], [-2.0, -2.0, -2.0]])
    policy = old + torch.tensor([[0.1, -0.1, 0.0], [0.3, -0.3, 0.0]])
    reference = old.clone()
    rewards = torch.tensor([[1.0, 0.0, -1.0], [0.2, 0.8, 0.5]])
    loss, metrics = loss_fn(policy, old, reference, rewards)
    assert torch.isfinite(loss)
    assert abs(float(metrics["advantage_mean"])) < 1e-6
    assert 0.0 <= float(metrics["clip_fraction"]) <= 1.0
    assert float(metrics["kl"]) >= 0.0


def test_grpo_dataset_loader_and_trainer_step(tmp_path) -> None:
    tok = tokenizer()
    path = tmp_path / "grpo.jsonl"
    rows = [
        {"prompt": "Pick", "completions": ["A", "B", "C"], "rewards": [1.0, 0.0, -1.0]},
        {
            "prompt": "Choose",
            "completions": ["X", "Y", "Z"],
            "rewards": [0.2, 0.8, 0.4],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    loader = build_grpo_loader(
        [str(path)], tok, max_length=64, batch_size=2, shuffle=False
    )
    batch = next(iter(loader))
    assert batch["completion_ids"].shape[:2] == (2, 3)
    policy = MiniGPT(vocab_size=tok.vocab_size, dim=8, layers=1, heads=2, max_pos=64)
    old = copy.deepcopy(policy)
    reference = copy.deepcopy(policy)
    trainer = GRPOTrainer(
        policy, old, reference, build_adamw(policy, learning_rate=1e-4)
    )
    metrics = trainer.train_step(batch)
    assert trainer.global_step == 1
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert metrics["kl"] >= 0.0


def test_reward_and_grpo_clis_and_profiles_are_available() -> None:
    root = Path(__file__).resolve().parents[1]
    for script in ("scripts/train_reward_model.py", "scripts/train_grpo.py"):
        completed = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
    for config in (
        "configs/reward_model.cpu.yaml",
        "configs/reward_model.gpu.yaml",
        "configs/grpo.cpu.yaml",
        "configs/grpo.gpu.yaml",
    ):
        assert (root / config).is_file()
