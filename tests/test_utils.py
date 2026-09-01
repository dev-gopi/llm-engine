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
