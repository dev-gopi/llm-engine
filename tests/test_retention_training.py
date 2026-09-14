import json
from pathlib import Path
import subprocess
import sys

import torch
import yaml

from model.gpt import MiniGPT
from tokenizer.bpe import BYTE_ENCODER
from tokenizer.encoder import DEFAULT_SPECIAL_TOKENS, Tokenizer
from training.checkpoint import save_checkpoint


def test_training_saves_initial_retention_baseline_before_first_update(tmp_path):
    pieces = [*DEFAULT_SPECIAL_TOKENS, *BYTE_ENCODER.values()]
    vocab = {piece: i for i, piece in enumerate(pieces)}
    tok = Tokenizer(vocab, special_tokens={piece: vocab[piece] for piece in DEFAULT_SPECIAL_TOKENS})
    tok.save(tmp_path / "tokenizer")
    config = dict(vocab_size=tok.vocab_size, hidden_size=8, layers=1, heads=2, max_position=128)
    model_path = tmp_path / "model.yaml"
    model_path.write_text(yaml.safe_dump(config))
    torch.manual_seed(17)
    model = MiniGPT.from_config(config)
    init = save_checkpoint(tmp_path / "init.pt", model, metadata={"tokenizer_fingerprint": tok.fingerprint})
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(json.dumps({"messages": [{"role": "user", "content": "Hi"},
                                               {"role": "assistant", "content": "Hello"}]}) + "\n")
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps({"category": "chat", "prompt": "Say hello.",
                                "expected": ["a deliberately longer answer than two tokens"], "match": "exact"}) + "\n")
    generation_best = tmp_path / "generation.pt"
    training = dict(batch_size=1, epochs=1, max_sequence_length=128, train_files=[str(dataset)],
                    validation_files=[str(dataset)], num_workers=0, learning_rate=0.0001,
                    mixed_precision="none", ema_decay=None, evaluate_every=1, log_every=0,
                    generation_evaluation=dict(enabled=True, cases=str(cases), max_tokens=2,
                                               output=str(tmp_path / "generation.json"),
                                               best_output=str(generation_best), weights="model",
                                               evaluate_at_start=True, preserve_passed=True))
    training_path = tmp_path / "training.yaml"
    training_path.write_text(yaml.safe_dump(training))
    completed = subprocess.run([
        sys.executable, "scripts/train.py", "--model-config", str(model_path),
        "--training-config", str(training_path), "--tokenizer", str(tmp_path / "tokenizer"),
        "--init-from", str(init), "--output", str(tmp_path / "latest.pt"),
        "--best-output", str(tmp_path / "loss.pt"), "--no-live-report",
        "--log-file", str(tmp_path / "train.log"), "--report-json", str(tmp_path / "report.json"),
    ], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    selected = torch.load(generation_best, weights_only=True)
    assert selected["step"] == 0
    assert selected["metadata"]["inference_only"]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(selected["model"][name], value, rtol=0, atol=0)
    assert torch.load(tmp_path / "latest.pt", weights_only=True)["step"] == 1
