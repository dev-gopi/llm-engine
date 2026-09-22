"""Tool authorization policy shared by native and MCP tools."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolAuthorization:
    allowed_tools: frozenset[str]
    require_approval: bool = True
    timeout_seconds: float = 30.0
    def authorize(self, tool_name: str, *, approved: bool = False) -> None:
        if tool_name not in self.allowed_tools: raise PermissionError(f"tool {tool_name!r} is not authorized")
        if self.require_approval and not approved: raise PermissionError("human approval is required")
        if self.timeout_seconds <= 0: raise ValueError("timeout_seconds must be positive")
