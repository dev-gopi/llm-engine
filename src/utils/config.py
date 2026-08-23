"""Validated YAML configuration loading."""

from pathlib import Path
from typing import Any

import yaml



def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"configuration root must be a mapping: {config_path}")
    return data
