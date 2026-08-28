import copy
import json
import subprocess
import sys
from pathlib import Path

import torch

from scripts.prepare_helpsteer_preferences import quality

from model.gpt import MiniGPT
from optim.adamw import build_adamw
from optim.scheduler import Scheduler
from post_training.dpo import DPOTrainer
from post_training.preference_data import build_preference_loader
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
    assert (root / "configs/dpo.v2.cpu.yaml").is_file()
    assert (root / "configs/dpo.v2.gpu.yaml").is_file()
