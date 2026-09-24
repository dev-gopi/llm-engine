"""Dependency-light Prometheus exposition and structured operation metrics."""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class _MetricState:
    counters: dict[tuple[str, tuple[tuple[str, str], ...]], float]
    gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float]


class Metrics:
    def __init__(self) -> None:
        self._state = _MetricState(defaultdict(float), {})
        self._lock = threading.Lock()

    @staticmethod
    def _key(name: str, labels: dict[str, str] | None) -> tuple[str, tuple[tuple[str, str], ...]]:
        return name, tuple(sorted((labels or {}).items()))

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        with self._lock:
            self._state.counters[self._key(name, labels)] += value

    def set(self, name: str, value: float, **labels: str) -> None:
        with self._lock:
            self._state.gauges[self._key(name, labels)] = value

    @contextmanager
    def time(self, name: str, **labels: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.set(name, time.perf_counter() - start, **labels)

    def prometheus(self) -> str:
        lines: list[str] = []
        with self._lock:
            items = list(self._state.counters.items()) + list(self._state.gauges.items())
        for (name, labels), value in sorted(items):
            label_text = ""
            if labels:
                encoded = ",".join(f'{k}="{v.replace(chr(34), "")}"' for k, v in labels)
                label_text = "{" + encoded + "}"
            lines.append(f"gopi_{name}{label_text} {value}")
        return "\n".join(lines) + ("\n" if lines else "")


METRICS = Metrics()
