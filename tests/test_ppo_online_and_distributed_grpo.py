import subprocess
import sys
from pathlib import Path

import torch

from inference.generator import GenerationResult
from model.gpt import MiniGPT
from post_training.ppo import ValueModel
from post_training.ppo_online import collect_ppo_rollouts
from tokenizer.bpe import BYTE_ENCODER
from tokenizer.encoder import DEFAULT_SPECIAL_TOKENS, Tokenizer


def tokenizer() -> Tokenizer:
    pieces = list(DEFAULT_SPECIAL_TOKENS) + list(BYTE_ENCODER.values())
    vocab = {piece: index for index, piece in enumerate(pieces)}
    return Tokenizer(
        vocab, special_tokens={piece: vocab[piece] for piece in DEFAULT_SPECIAL_TOKENS}
    )


class FakeGenerator:
    def generate(self, prompt, **kwargs):
        return GenerationResult(
            text="4", token_ids=(1,), prompt_tokens=1, finish_reason="stop"
        )


class ZeroScorer:
    def score(self, prompt, completion, *, normalized=False):
        return 0.0


def test_collect_ppo_rollouts_builds_frozen_sequence_statistics() -> None:
    tok = tokenizer()
    policy = MiniGPT(vocab_size=tok.vocab_size, dim=16, layers=1, heads=2, max_pos=64)
    value = ValueModel(
        MiniGPT(vocab_size=tok.vocab_size, dim=16, layers=1, heads=2, max_pos=64), 16
    )
    rollout, failures = collect_ppo_rollouts(
        [{"prompt": "2+2?", "expected_answer": "4"}],
        generator=FakeGenerator(),
        tokenizer=tok,
        old_policy=policy,
        value_model=value,
        scorer=ZeroScorer(),
        device=torch.device("cpu"),
        max_sequence_length=64,
        reward_model_weight=0.0,
        exact_match_weight=1.0,
    )
    assert failures == []
    assert rollout["token_ids"].shape[0] == 1
    assert rollout["old_log_probs"].shape == (1,)
    assert rollout["old_values"].shape == (1,)
    assert rollout["rewards"].tolist() == [1.0]
    assert rollout["mask"].any()


def test_ppo_cli_profiles_and_alignment_registry() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "scripts/train_ppo.py", "--help"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert (root / "configs/ppo.cpu.yaml").is_file()
    assert (root / "configs/ppo.gpu.yaml").is_file()
    alignment = (root / "configs/alignment.methods.yaml").read_text(encoding="utf-8")
    assert "ppo" in alignment.split("methods:", 1)[1].splitlines()[0]
    assert "planned_methods: []" in alignment


def test_online_grpo_profiles_enable_ddp_strategy() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ("configs/grpo.online.cpu.yaml", "configs/grpo.online.gpu.yaml"):
        text = (root / name).read_text(encoding="utf-8")
        assert "distributed_strategy: ddp" in text
    source = (root / "scripts/train_grpo_online.py").read_text(encoding="utf-8")
    assert "shard_prompts" in source
    assert "aggregate_rollouts" in source
    assert "DistributedTrainer.wrap" in source
