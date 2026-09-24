"""REST and WebSocket serving for Gopi."""

from typing import TYPE_CHECKING, Any

from .runtime import BackendGeneration, BackendStreamEvent, GenerationBackend

if TYPE_CHECKING:
    from .api import ServingSettings, app, create_app

__all__ = [
    "BackendGeneration",
    "BackendStreamEvent",
    "GenerationBackend",
    "ServingSettings",
    "app",
    "create_app",
]


def __getattr__(name: str) -> Any:
    """Lazily expose API objects without creating schema import cycles."""
    if name in {"ServingSettings", "app", "create_app"}:
        from . import api

        return getattr(api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
