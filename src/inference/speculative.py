"""Serving-safe speculative decoding contracts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class SpeculativeConfig:
    max_draft_tokens: int = 5
    acceptance_temperature: float = 1.0
    draft_model_id: str | None = None
    enabled: bool = False

    def __post_init__(self) -> None:
        if self.max_draft_tokens < 1:
            raise ValueError("max_draft_tokens must be positive")
        if self.acceptance_temperature <= 0:
            raise ValueError("acceptance_temperature must be positive")
        if self.enabled and not self.draft_model_id:
            raise ValueError("enabled speculative decoding requires draft_model_id")


class DraftModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, object] = {}

    def register(self, model_id: str, model: object) -> None:
        if not model_id.strip() or model_id in self._models:
            raise ValueError("draft model id must be unique and non-empty")
        self._models[model_id] = model

    def replace(self, model_id: str, model: object) -> None:
        if model_id not in self._models:
            raise KeyError(model_id)
        self._models[model_id] = model

    def remove(self, model_id: str) -> None:
        self._models.pop(model_id, None)

    def get(self, model_id: str) -> object:
        return self._models[model_id]

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._models))


def speculative_accept(
    proposed: Sequence[T],
    target_accept: Callable[[T, int], bool],
) -> tuple[list[T], int]:
    """Accept a contiguous prefix according to the target model.

    The target callback owns probability/logit computation; this helper only
    implements the deterministic lifecycle and never invents acceptance scores.
    """
    accepted: list[T] = []
    for index, token in enumerate(proposed):
        if not target_accept(token, index):
            return accepted, index
        accepted.append(token)
    return accepted, len(proposed)
