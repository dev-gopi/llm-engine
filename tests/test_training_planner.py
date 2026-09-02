import json
import subprocess
import sys
from pathlib import Path

import pytest

from training.planner import optimizer_steps_for_epochs, plan_training


MODEL = {
    "vocab_size": 1000,
    "hidden_size": 64,
    "layers": 2,
    "heads": 4,
    "max_position": 128,
    "ffn_hidden_size": 256,
    "gradient_checkpointing": True,
}


def test_optimizer_steps_count_partial_accumulation_per_epoch() -> None:
    assert optimizer_steps_for_epochs(5, epochs=2, accumulation_steps=4) == 4
    assert optimizer_steps_for_epochs(8, epochs=2, accumulation_steps=4) == 4


def test_training_planner_estimates_steps_flops_runtime_cost_and_memory() -> None:
    plan = plan_training(
        MODEL,
        {
            "batch_size": 2,
            "max_sequence_length": 128,
            "gradient_accumulation_steps": 4,
            "distributed_strategy": "ddp",
            "ema_decay": 0.999,
        },
        training_tokens=10_000,
        gpus=2,
        hardware_tflops=10,
        utilization=0.5,
        gpu_memory_gib=1,
        hourly_cost_per_gpu=2,
    )
    assert plan.tokens_per_optimizer_step == 2048
    assert plan.optimizer_steps == 5
    assert plan.training_flops == 6 * plan.parameters * 10_000
    assert plan.estimated_hours is not None and plan.estimated_hours > 0
    assert plan.estimated_cost == pytest.approx(plan.estimated_hours * 4)
    assert plan.fits_memory


def test_fsdp_planner_shards_model_state_but_not_activations() -> None:
    common = {"batch_size": 1, "max_sequence_length": 64}
    ddp = plan_training(MODEL, {**common, "distributed_strategy": "ddp"}, training_tokens=1000, gpus=4)
    fsdp = plan_training(MODEL, {**common, "distributed_strategy": "fsdp"}, training_tokens=1000, gpus=4)
    assert fsdp.model_state_gib_per_gpu < ddp.model_state_gib_per_gpu
    assert fsdp.activation_gib_per_gpu == ddp.activation_gib_per_gpu


def test_training_planner_cli_emits_json_without_allocating_model() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable, "scripts/plan_training.py",
            "--model-config", "configs/model.gpu.yaml",
            "--training-config", "configs/pretraining.gpu.yaml",
            "--training-tokens", "1000000", "--gpus", "1",
            "--hardware-tflops", "10", "--gpu-memory-gib", "4",
        ],
        cwd=root, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["training_tokens"] == 1_000_000
    assert payload["token_estimate"]["source"] == "explicit"
    assert payload["assumptions"]["estimate_only"] is True


def test_training_planner_cli_can_enforce_feasibility_constraints() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable, "scripts/plan_training.py",
            "--model-config", "configs/text/model.future.1b.yaml",
            "--training-config", "configs/text/pretraining.future.fsdp.yaml",
            "--training-tokens", "1000000", "--gpu-memory-gib", "0.01",
            "--require-fit",
        ],
        cwd=root, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert not payload["constraints"]["passed"]
    assert payload["constraints"]["violations"]
