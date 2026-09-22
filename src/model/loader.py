"""Safe model loader abstractions used by protected lifecycle operations."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LoadedModel:
    backend: Any
    version: str


class ModelLoader:
    """Allow lifecycle code to load only preconfigured model factories."""

    def __init__(self, factory: Callable[[], tuple[Any, str]]) -> None:
        if not callable(factory):
            raise TypeError("model loader factory must be callable")
        self.factory = factory

    def load(self) -> LoadedModel:
        backend, version = self.factory()
        if backend is None or not version:
            raise RuntimeError("model factory returned an invalid backend/version")
        return LoadedModel(backend=backend, version=str(version))
