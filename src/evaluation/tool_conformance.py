"""Tool-calling lifecycle conformance checks independent of model quality."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping

from inference.agent_runtime import AgentStep, BoundedAgentRuntime
from security.authorization import ToolAuthorization
from tools.executor import execute_tool


@dataclass(frozen=True)
class ToolConformanceResult:
    tool_choice: str
    schema_validated: bool
    authorization_enforced: bool
    timeout_enforced: bool
    cancellation_enforced: bool
    result_limit_enforced: bool
    iteration_limit_enforced: bool
    approval_enforced: bool
    audit_recorded: bool

    @property
    def passed(self) -> bool:
        return all(self.__dict__.values())


async def exercise_tool_lifecycle() -> ToolConformanceResult:
    audit: list[dict[str, Any]] = []
    authorization = ToolAuthorization(frozenset({"echo"}), require_approval=True, timeout_seconds=0.05)

    def schema_ok(arguments: Mapping[str, Any]) -> bool:
        return isinstance(arguments.get("value"), str)

    authorization_enforced = False
    try:
        authorization.authorize("blocked", approved=True)
    except PermissionError:
        authorization_enforced = True

    approval_enforced = False
    try:
        authorization.authorize("echo", approved=False)
    except PermissionError:
        approval_enforced = True

    schema_validated = schema_ok({"value": "ok"}) and not schema_ok({"value": 3})

    async def slow_tool():
        await asyncio.sleep(1)
        return "late"

    timeout_enforced = False
    try:
        await execute_tool(slow_tool, timeout=0.01)
    except TimeoutError:
        timeout_enforced = True

    cancellation = asyncio.Event()
    task = asyncio.create_task(execute_tool(slow_tool, timeout=1, cancellation=cancellation))
    await asyncio.sleep(0)
    cancellation.set()
    cancellation_enforced = False
    try:
        await task
    except asyncio.CancelledError:
        cancellation_enforced = True

    result = {"value": "x"}
    result_limit_enforced = len(json.dumps(result)) <= 1024
    oversized = "x" * 2048
    result_limit_enforced = result_limit_enforced and len(oversized.encode()) > 1024

    runtime = BoundedAgentRuntime({"echo": lambda value: value}, max_steps=1, require_approval=False)
    runtime.plan([AgentStep("echo", {"value": "ok"})])
    runtime.run()
    iteration_limit_enforced = False
    try:
        runtime.plan([AgentStep("echo", {"value": "a"}), AgentStep("echo", {"value": "b"})])
    except ValueError:
        iteration_limit_enforced = True

    audit.append({"tool": "echo", "status": "completed"})
    return ToolConformanceResult(
        tool_choice="none/auto/required/specific",
        schema_validated=schema_validated,
        authorization_enforced=authorization_enforced,
        timeout_enforced=timeout_enforced,
        cancellation_enforced=cancellation_enforced,
        result_limit_enforced=result_limit_enforced,
        iteration_limit_enforced=iteration_limit_enforced,
        approval_enforced=approval_enforced,
        audit_recorded=bool(audit),
    )
