from __future__ import annotations

import pytest
import torch
from torch import nn

from training.parallel_runtime import apply_model_parallelism, resolve_parallel_degrees
from training.trainer import Trainer


def test_parallel_degrees_infer_data_parallel_from_world_size():
    degrees = resolve_parallel_degrees(
        {"parallelism": {"tp": 2, "pp": 2, "ep": 2}}, world_size=16
    )
    assert degrees.data == 2
    assert degrees.tensor == 2
    assert degrees.pipeline == 2
    assert degrees.expert == 2
    assert degrees.world_size == 16


def test_parallel_degrees_reject_incompatible_world_size():
    with pytest.raises(ValueError, match="not divisible"):
        resolve_parallel_degrees({"parallelism": {"tp": 2, "pp": 2}}, world_size=6)


def test_single_rank_parallelism_is_noop_but_attaches_mesh_marker():
    model = nn.Linear(2, 2)
    assert apply_model_parallelism(model, None) is model
    assert model.parallel_mesh is None


class _FakeDeepSpeedEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 8)
        self.backward_calls = 0
        self.step_calls = 0

    def forward(self, token_ids, attention_mask=None, **kwargs):
        x = torch.nn.functional.one_hot(token_ids, num_classes=4).float()
        return self.proj(x)

    def backward(self, loss):
        self.backward_calls += 1
        loss.backward()

    def step(self):
        self.step_calls += 1

    def is_gradient_accumulation_boundary(self):
        return self.step_calls % 2 == 1


def test_trainer_uses_deepspeed_backward_and_step():
    engine = _FakeDeepSpeedEngine()
    optimizer = torch.optim.AdamW(engine.parameters(), lr=1e-3)
    trainer = Trainer(
        engine,
        optimizer,
        device="cpu",
        gradient_accumulation_steps=2,
        deepspeed_engine=True,
    )
    batch = {
        "input_ids": torch.tensor([[0, 1, 2]]),
        "labels": torch.tensor([[0, 1, 2]]),
    }
    trainer.train_step(batch)
    assert engine.backward_calls == 1
    assert engine.step_calls == 1
