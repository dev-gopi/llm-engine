import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from inference.generator import GenerationResult
from model.gpt import MiniGPT
from post_training.grpo_data import build_grpo_loader_from_records
from post_training.online_grpo import OnlineRolloutGenerator, RolloutConfig, RolloutReplayBuffer, load_prompt_records
from post_training.reward_model import RewardModel
from post_training.reward_scoring import RewardCalibration, RewardScorer, fit_reward_calibration
from tokenizer.bpe import BYTE_ENCODER
from tokenizer.encoder import DEFAULT_SPECIAL_TOKENS, Tokenizer


def tokenizer() -> Tokenizer:
    pieces = list(DEFAULT_SPECIAL_TOKENS) + list(BYTE_ENCODER.values())
    vocab = {piece: index for index, piece in enumerate(pieces)}
    return Tokenizer(vocab, special_tokens={piece: vocab[piece] for piece in DEFAULT_SPECIAL_TOKENS})


def test_reward_calibration_fits_normalization_and_pairwise_temperature(tmp_path) -> None:
    calibration, metrics = fit_reward_calibration([3.0, 2.0, 1.5], [0.0, -1.0, 0.5])
    assert calibration.std > 0
    assert calibration.temperature > 0
    assert metrics["pairwise_accuracy"] == 1.0
    assert 0.0 <= metrics["pairwise_brier"] <= 1.0
    path = tmp_path / "calibration.json"
    calibration.save(path, metadata={"checkpoint": "reward.pt"})
    loaded = RewardCalibration.load(path)
    assert loaded == calibration
    assert loaded.normalize(loaded.mean) == pytest.approx(0.0)
    assert loaded.pairwise_probability(2.0, 1.0) > 0.5


def test_reward_scorer_batches_and_normalizes() -> None:
    tok = tokenizer()
    backbone = MiniGPT(vocab_size=tok.vocab_size, dim=8, layers=1, heads=2, max_pos=64)
    model = RewardModel(backbone, 8)
    calibration = RewardCalibration(mean=0.0, std=2.0, temperature=1.0)
    scorer = RewardScorer(model, tok, max_length=64, calibration=calibration)
    raw = scorer.score_many([("Q", "A"), ("Q2", "A2")], batch_size=2)
    normalized = scorer.score_many([("Q", "A"), ("Q2", "A2")], batch_size=2, normalized=True)
    assert len(raw) == len(normalized) == 2
    assert all(torch.isfinite(torch.tensor(raw)))
    assert normalized == pytest.approx([value / 2.0 for value in raw])


class FakeGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt, **kwargs):
        self.calls += 1
        text = f"candidate-{self.calls}"
        return GenerationResult(text=text, token_ids=(self.calls,), prompt_tokens=1, finish_reason="stop")


class FakeScorer:
    def score(self, prompt, completion, *, normalized=False):
        return float(completion.rsplit("-", 1)[1])


def test_online_rollout_generation_produces_grpo_compatible_groups() -> None:
    rollout = OnlineRolloutGenerator(
        FakeGenerator(), FakeScorer(),
        RolloutConfig(group_size=3, max_tokens=8, reward_model_weight=1.0, seed=7),
    )
    groups, failures = rollout.generate([{"prompt": "Choose"}])
    assert failures == []
    assert len(groups) == 1
    assert groups[0]["completions"] == ["candidate-1", "candidate-2", "candidate-3"]
    assert groups[0]["rewards"] == [1.0, 2.0, 3.0]
    tok = tokenizer()
    loader = build_grpo_loader_from_records(groups, tok, max_length=64, batch_size=1, shuffle=False)
    batch = next(iter(loader))
    assert batch["completion_ids"].shape[1] == 3


def test_online_rollout_exact_match_signal_without_reward_model() -> None:
    rollout = OnlineRolloutGenerator(
        FakeGenerator(), None,
        RolloutConfig(group_size=2, max_tokens=8, reward_model_weight=0.0, exact_match_weight=1.0),
    )
    # candidate-2 is the expected answer, yielding a non-constant reward group.
    group = rollout.generate_group({"prompt": "Q", "expected_answer": "candidate-2"})
    assert group is not None
    assert group["rewards"] == [0.0, 1.0]


def test_rollout_generator_records_faults_instead_of_aborting_all_prompts() -> None:
    class BrokenOnce(FakeGenerator):
        def generate(self, prompt, **kwargs):
            if "bad" in prompt:
                raise RuntimeError("boom")
            return super().generate(prompt, **kwargs)

    rollout = OnlineRolloutGenerator(BrokenOnce(), FakeScorer(), RolloutConfig(group_size=2))
    groups, failures = rollout.generate([{"prompt": "bad"}, {"prompt": "good"}])
    assert len(groups) == 1
    assert len(failures) == 1
    assert failures[0]["reason"] == "RuntimeError"


def test_replay_buffer_is_bounded_and_persistent(tmp_path) -> None:
    buffer = RolloutReplayBuffer(2)
    first = {"prompt": "1", "completions": ["a", "b"], "rewards": [1, 0]}
    second = {"prompt": "2", "completions": ["a", "b"], "rewards": [1, 0]}
    third = {"prompt": "3", "completions": ["a", "b"], "rewards": [1, 0]}
    buffer.extend_current([first, second])
    values = buffer.extend_current([third])
    assert [item["prompt"] for item in values] == ["2", "3"]
    path = tmp_path / "replay.jsonl"
    buffer.save(path)
    restored = RolloutReplayBuffer(2)
    assert restored.load(path) == 2
    assert len(restored) == 2


def test_prompt_loader_accepts_string_or_object_jsonl(tmp_path) -> None:
    path = tmp_path / "prompts.jsonl"
    path.write_text(json.dumps("hello") + "\n" + json.dumps({"prompt": "world"}) + "\n", encoding="utf-8")
    assert load_prompt_records([path]) == [{"prompt": "hello"}, {"prompt": "world"}]


def test_online_grpo_reward_scoring_clis_and_profiles_are_available() -> None:
    root = Path(__file__).resolve().parents[1]
    for script in (
        "scripts/calibrate_reward_model.py",
        "scripts/score_rewards.py",
        "scripts/generate_grpo_rollouts.py",
        "scripts/train_grpo_online.py",
    ):
        completed = subprocess.run([sys.executable, script, "--help"], cwd=root, text=True, capture_output=True, check=False)
        assert completed.returncode == 0, completed.stderr
    for config in ("configs/grpo.online.cpu.yaml", "configs/grpo.online.gpu.yaml"):
        assert (root / config).is_file()
