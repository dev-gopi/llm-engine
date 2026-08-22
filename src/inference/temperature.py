"""Logit temperature scaling."""

import torch
from torch import Tensor


def apply_temperature(logits: Tensor, temperature: float) -> Tensor:
    if not isinstance(logits, Tensor) or not logits.is_floating_point():
        raise TypeError("logits must be a floating-point tensor")
    if not torch.isfinite(torch.tensor(temperature)) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    if temperature == 0:
        return logits
    return logits / temperature
