import logging
import random

import numpy as np
import pytest
import torch

from utils.config import load_yaml
from utils.device import resolve_device
from utils.logger import configure_logging, get_logger
from utils.seed import set_seed


def test_load_yaml_validates_root(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("model:\n  hidden_size: 32\n", encoding="utf-8")
    assert load_yaml(config)["model"]["hidden_size"] == 32
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("- item\n", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be a mapping"):
        load_yaml(invalid)


def test_seed_controls_python_numpy_and_torch() -> None:
    set_seed(123)
    first = (random.random(), np.random.rand(), torch.rand(2))
    set_seed(123)
    second = (random.random(), np.random.rand(), torch.rand(2))
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_device_and_logger_helpers() -> None:
    assert resolve_device("cpu") == torch.device("cpu")
    configure_logging("WARNING")
    logger = get_logger("tests")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "llm_engine.tests"


def test_invalid_seed_and_log_level() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        set_seed(-1)
    with pytest.raises(ValueError, match="logging level"):
        configure_logging("not-a-level")


def test_logging_can_append_to_a_report_file(tmp_path) -> None:
    destination = tmp_path / "training.log"
    configure_logging("INFO", log_file=destination)
    logger = get_logger("file-test")
    logger.info("report marker")
    for handler in logger.parent.handlers if logger.parent else ():
        handler.flush()
    assert "report marker" in destination.read_text(encoding="utf-8")


def test_yaml_inheritance_merges_nested_defaults_and_replaces_lists(tmp_path):
    (tmp_path / "base.yaml").write_text("runtime:\n  device: cpu\n  count: 2\nfiles: [a, b]\n")
    (tmp_path / "second.yaml").write_text("runtime:\n  count: 4\n")
    child = tmp_path / "child.yaml"
    child.write_text("extends: [base.yaml, second.yaml]\nruntime:\n  device: cuda\nfiles: [c]\n")
    assert load_yaml(child) == {"runtime": {"device": "cuda", "count": 4}, "files": ["c"]}
    assert load_yaml(tmp_path / "base.yaml")["runtime"]["count"] == 2


def test_yaml_inheritance_rejects_cycles_and_invalid_parent(tmp_path):
    config = tmp_path / "cycle.yaml"
    config.write_text("extends: cycle.yaml\n")
    with pytest.raises(ValueError, match="cycle"):
        load_yaml(config)
    config.write_text("extends: 123\n")
    with pytest.raises(ValueError, match="extends"):
        load_yaml(config)


def test_cli_overrides_config_and_preserves_explicit_zero():
    from argparse import Namespace
    from pathlib import Path
    from utils.config import apply_cli_defaults
    args = Namespace(count=0, output=None, device=None)
    apply_cli_defaults(args, {"count": 5, "output": "custom.pt"},
                       {"count": 2, "output": Path("default.pt"), "device": "cpu"})
    assert args.count == 0
    assert args.output == Path("custom.pt")
    assert args.device == "cpu"


def test_every_repository_yaml_loads_with_shared_defaults():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for path in (root / "configs").rglob("*.yaml"):
        assert isinstance(load_yaml(path), dict)
    fine = load_yaml(root / "configs/finetuning.gpu.yaml")
    assert fine["runtime"]["report_telemetry_points"] == 3600
    assert fine["runtime"]["tokenizer"] == "data/tokenizer-finetuning"
