"""Bounded tool execution with authorization, schema, cancellation, audit and size controls."""
from __future__ import annotations

import asyncio
import json
from typing import Awaitable, Callable, Any, Mapping

from security.authorization import ToolAuthorization


async def execute_tool(
    tool: Callable[..., Awaitable[Any]],
    *args,
    timeout: float = 30.0,
    cancellation: asyncio.Event | None = None,
    **kwargs,
) -> Any:
    task = asyncio.create_task(tool(*args, **kwargs))
    try:
        if cancellation is None:
            return await asyncio.wait_for(task, timeout=timeout)
        cancel_wait = asyncio.create_task(cancellation.wait())
        done, pending = await asyncio.wait({task, cancel_wait}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        if cancel_wait in done and cancellation.is_set():
            task.cancel(); await asyncio.gather(task, return_exceptions=True); raise asyncio.CancelledError
        cancel_wait.cancel(); await asyncio.gather(cancel_wait, return_exceptions=True)
        if task not in done:
            task.cancel(); await asyncio.gather(task, return_exceptions=True); raise TimeoutError("tool execution timed out")
        return task.result()
    except BaseException:
        if not task.done():
            task.cancel(); await asyncio.gather(task, return_exceptions=True)
        raise


class ControlledToolExecutor:
    """Single lifecycle for native and MCP-style tools.

    Authorization and JSON-schema validation happen before the callable is
    started. Results are bounded before being returned to the agent runtime.
    """

    def __init__(
        self,
        tools: Mapping[str, Callable[..., Awaitable[Any]]],
        *,
        authorization: ToolAuthorization,
        schemas: Mapping[str, Mapping[str, Any]] | None = None,
        max_result_bytes: int = 1_048_576,
        audit: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if max_result_bytes < 1:
            raise ValueError("max_result_bytes must be positive")
        self.tools = dict(tools)
        self.authorization = authorization
        self.schemas = dict(schemas or {})
        self.max_result_bytes = max_result_bytes
        self.audit = audit

    @staticmethod
    def _validate_schema(arguments: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
        from jsonschema import Draft202012Validator
        try:
            validator = Draft202012Validator(dict(schema))
            validator.validate(dict(arguments))
        except Exception as error:
            raise ValueError(f"tool arguments failed JSON Schema validation: {error}") from error

    def _record(self, event: dict[str, Any]) -> None:
        if self.audit is not None:
            self.audit(dict(event))

    async def execute(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        approved: bool = False,
        cancellation: asyncio.Event | None = None,
    ) -> Any:
        args = dict(arguments or {})
        self.authorization.authorize(name, approved=approved)
        if name not in self.tools:
            raise PermissionError(f"tool {name!r} is not registered")
        schema = self.schemas.get(name)
        if schema is not None:
            self._validate_schema(args, schema)
        self._record({"event": "tool_started", "tool": name})
        try:
            result = await execute_tool(
                self.tools[name], timeout=self.authorization.timeout_seconds,
                cancellation=cancellation, **args,
            )
            encoded = json.dumps(result, ensure_ascii=False, default=str).encode("utf-8")
            if len(encoded) > self.max_result_bytes:
                raise ValueError("tool result exceeds configured size limit")
            self._record({"event": "tool_completed", "tool": name, "result_bytes": len(encoded)})
            return result
        except BaseException as error:
            self._record({"event": "tool_failed", "tool": name, "error": type(error).__name__})
            raise
