import pytest
import json
from pathlib import Path
import subprocess
import sys

from training.sweeps import build_sweep, sweep_manifest


def test_build_sweep_is_deterministic_and_preserves_base_config() -> None:
    base = {"batch_size": 2, "learning_rate": 0.001, "nested": {"kept": True}}
    trials = build_sweep(base, {"learning_rate": [0.001, 0.0005], "lr_schedule": ["cosine", "linear"]})
    assert [trial.name for trial in trials] == ["trial-001", "trial-002", "trial-003", "trial-004"]
    assert trials[0].overrides == {"learning_rate": 0.001, "lr_schedule": "cosine"}
    assert trials[-1].config["learning_rate"] == 0.0005
    assert trials[-1].config["nested"] == {"kept": True}
    assert base == {"batch_size": 2, "learning_rate": 0.001, "nested": {"kept": True}}
    assert sweep_manifest(trials)["format_version"] == 1


@pytest.mark.parametrize("axes", [{}, {"unknown": [1]}, {"batch_size": []}, {"optimizer": ["sgd"]}])
def test_build_sweep_rejects_invalid_or_unsupported_axes(axes) -> None:
    with pytest.raises(ValueError):
        build_sweep({}, axes)


def test_sweep_cli_prints_a_dry_run_manifest(tmp_path) -> None:
    config_path = tmp_path / "training.yaml"
    axes_path = tmp_path / "axes.yaml"
    config_path.write_text("batch_size: 2\nlearning_rate: 0.001\n", encoding="utf-8")
    axes_path.write_text("learning_rate: [0.001, 0.0005]\n", encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/run_sweep.py", "--training-config", str(config_path), "--axes", str(axes_path)],
        cwd=root, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["trial_count"] == 2
