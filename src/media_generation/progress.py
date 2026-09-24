"""Small progress/cancellation primitives usable by CLI and serving layers."""
from __future__ import annotations
from dataclasses import dataclass
from threading import Event
from typing import Callable

ProgressCallback = Callable[[int, int], None]

@dataclass
class CancellationToken:
    _event: Event
    @classmethod
    def create(cls) -> "CancellationToken":
        return cls(Event())
    def cancel(self) -> None:
        self._event.set()
    @property
    def cancelled(self) -> bool:
        return self._event.is_set()
