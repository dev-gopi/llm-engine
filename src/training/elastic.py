"""Coordinated preemption handling for torchrun and elastic workers."""

from __future__ import annotations

import signal
import threading

import torch
import torch.distributed as dist


class PreemptionCoordinator:
    """Turn termination signals into a collective, checkpointable stop request."""

    def __init__(self) -> None:
        self._requested = threading.Event()
        self._previous: dict[signal.Signals, object] = {}

    def install(self) -> None:
        for name in ("SIGTERM", "SIGUSR1"):
            signum = getattr(signal, name, None)
            if signum is not None:
                self._previous[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle)

    def restore(self) -> None:
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)
        self._previous.clear()

    def request(self) -> None:
        self._requested.set()

    def should_stop(self, device: torch.device | str = "cpu") -> bool:
        flag = torch.tensor(int(self._requested.is_set()), device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def _handle(self, _signum, _frame) -> None:
        self.request()
