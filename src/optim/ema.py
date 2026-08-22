"""Exponential moving average of trainable model parameters."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch
from torch import nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0 < decay < 1:
            raise ValueError("decay must satisfy 0 < decay < 1")
        self.decay = decay
        self.num_updates = 0
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.is_floating_point()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        parameters = dict(model.named_parameters())
        for name, average in self.shadow.items():
            if name not in parameters:
                raise ValueError(f"model is missing EMA parameter {name}")
            average.lerp_(parameters[name].detach(), 1.0 - self.decay)
        self.num_updates += 1

    @contextmanager
    def average_parameters(self, model: nn.Module):
        parameters = dict(model.named_parameters())
        backup = {name: parameters[name].detach().clone() for name in self.shadow}
        try:
            with torch.no_grad():
                for name, average in self.shadow.items():
                    parameters[name].copy_(average)
            yield model
        finally:
            with torch.no_grad():
                for name, value in backup.items():
                    parameters[name].copy_(value)

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "num_updates": self.num_updates, "shadow": self.shadow}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.decay = float(state["decay"])
        self.num_updates = int(state["num_updates"])
        self.shadow = {name: value.detach().clone() for name, value in state["shadow"].items()}
