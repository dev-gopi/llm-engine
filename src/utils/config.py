"""Validated YAML loading with optional shared defaults."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML; `extends` paths are relative to the containing YAML file.

    Parent files merge left to right, then the child wins. Mappings merge
    recursively; lists and scalars replace. Ordinary data paths are unchanged.
    """
    return _load_yaml(Path(path).expanduser(), ())


def _load_yaml(path: Path, ancestors: tuple[Path, ...]) -> dict[str, Any]:
    config_path = path.resolve()
    if config_path in ancestors:
        raise ValueError(f"configuration inheritance cycle: {config_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"configuration root must be a mapping: {config_path}")
    parents = data.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list) or any(not isinstance(p, str) or not p for p in parents):
        raise ValueError("extends must be a path or a list of paths")
    merged: dict[str, Any] = {}
    for parent in parents:
        parent_path = Path(parent).expanduser()
        if not parent_path.is_absolute():
            parent_path = config_path.parent / parent_path
        merged = _merge(merged, _load_yaml(parent_path, (*ancestors, config_path)))
    return _merge(merged, data)


def apply_cli_defaults(args, config: dict[str, Any], defaults: dict[str, Any]) -> None:
    """Fill omitted CLI options from a mapping, then typed fallback defaults."""
    if not isinstance(config, dict):
        raise ValueError("runtime configuration must be a mapping")
    for name, default in defaults.items():
        if getattr(args, name, None) is None:
            value = config.get(name, default)
            if value is None:
                value = default
            setattr(args, name, type(default)(value))
