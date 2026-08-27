"""Async MCP client with stdio transport and modern/legacy negotiation."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODERN_PROTOCOL_VERSION = "2026-07-28"
LEGACY_PROTOCOL_VERSION = "2025-06-18"


class MCPError(RuntimeError):
    """Base error raised by the MCP client."""


class MCPProtocolError(MCPError):
    """JSON-RPC or MCP protocol error."""

    def __init__(self, message: str, *, code: int | None = None, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


@dataclass(frozen=True)
class MCPTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    title: str | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "MCPTool":
        name = payload.get("name")
        schema = payload.get("inputSchema", {})
        if not isinstance(name, str) or not name or not isinstance(schema, dict):
            raise MCPProtocolError("server returned an invalid tool definition")
        return cls(
            name=name,
            title=payload.get("title") if isinstance(payload.get("title"), str) else None,
            description=str(payload.get("description", "")),
            input_schema=dict(schema),
        )


class MCPClient:
    """Launch and communicate with one MCP server over newline-delimited stdio."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float = 30.0,
        client_name: str = "gopi-llm-engine",
        client_version: str = "0.1.0",
        protocol: str = "auto",
        max_message_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        if isinstance(command, (str, bytes)) or not command or any(not isinstance(part, str) or not part for part in command):
            raise ValueError("MCP command must be a non-empty sequence of strings")
        if timeout <= 0:
            raise ValueError("MCP timeout must be positive")
        if protocol not in {"auto", "modern", "legacy"}:
            raise ValueError("MCP protocol must be auto, modern, or legacy")
        if max_message_bytes < 1024:
            raise ValueError("MCP max_message_bytes must be at least 1024")
        self.command = tuple(command)
        self.cwd = Path(cwd) if cwd is not None else None
        self.env = dict(env or {})
        self.timeout = float(timeout)
        self.client_info = {"name": client_name, "version": client_version}
        self.protocol = protocol
        self.max_message_bytes = int(max_message_bytes)
        self.protocol_version: str | None = None
        self.server_info: dict[str, Any] = {}
        self.server_capabilities: dict[str, Any] = {}
        self.process: asyncio.subprocess.Process | None = None
        self._next_id = 1
        self._lock = asyncio.Lock()
        self._stderr_task: asyncio.Task | None = None
        self.stderr: list[str] = []

    async def __aenter__(self) -> "MCPClient":
        await self.start()
        return self

    async def __aexit__(self, *_exc_info) -> None:
        await self.close()

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.returncode is None

    async def start(self) -> None:
        if self.running:
            return
        environment = os.environ.copy()
        environment.update(self.env)
        try:
            self.process = await asyncio.create_subprocess_exec(
                *self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,
                env=environment,
                limit=self.max_message_bytes,
            )
        except OSError as error:
            raise MCPError(f"failed to start MCP server {self.command[0]!r}: {error}") from error
        self._stderr_task = asyncio.create_task(self._read_stderr())
        try:
            await self._negotiate()
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
            try:
                await process.stdin.wait_closed()
            except (BrokenPipeError, ConnectionResetError):
                pass
        if process.returncode is None:
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
        if self._stderr_task is not None:
            await self._stderr_task
            self._stderr_task = None

    async def list_tools(self) -> list[MCPTool]:
        tools: list[MCPTool] = []
        cursor: str | None = None
        while True:
            params = {"cursor": cursor} if cursor else {}
            result = await self.request("tools/list", params)
            payloads = result.get("tools")
            if not isinstance(payloads, list):
                raise MCPProtocolError("tools/list result does not contain a tools list")
            tools.extend(MCPTool.from_payload(payload) for payload in payloads if isinstance(payload, Mapping))
            cursor = result.get("nextCursor")
            if not isinstance(cursor, str) or not cursor:
                return tools

    async def call_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if not name.strip():
            raise ValueError("tool name cannot be empty")
        result = await self.request("tools/call", {"name": name, "arguments": dict(arguments or {})})
        if not isinstance(result.get("content", []), list):
            raise MCPProtocolError("tools/call result contains invalid content")
        return result

    async def request(self, method: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if not self.running or self.process is None:
            raise MCPError("MCP client is not started")
        async with self._lock:
            request_id = self._next_id
            self._next_id += 1
            payload: dict[str, Any] = {
                "jsonrpc": "2.0", "id": request_id, "method": method, "params": dict(params or {}),
            }
            if self.protocol_version == MODERN_PROTOCOL_VERSION:
                payload["_meta"] = {
                    "io.modelcontextprotocol/protocolVersion": self.protocol_version,
                    "io.modelcontextprotocol/clientInfo": self.client_info,
                    "io.modelcontextprotocol/clientCapabilities": {},
                }
            await self._write(payload)
            response = await self._read_response(request_id)
        error = response.get("error")
        if isinstance(error, Mapping):
            raise MCPProtocolError(
                str(error.get("message", "MCP request failed")),
                code=error.get("code") if isinstance(error.get("code"), int) else None,
                data=error.get("data"),
            )
        result = response.get("result")
        if not isinstance(result, dict):
            raise MCPProtocolError("MCP response does not contain an object result")
        return result

    async def _negotiate(self) -> None:
        if self.protocol == "legacy":
            await self._initialize_legacy()
            return
        self.protocol_version = MODERN_PROTOCOL_VERSION
        if self.protocol == "modern":
            return
        try:
            discovered = await self.request("server/discover")
        except (MCPError, asyncio.TimeoutError):
            await self._initialize_legacy()
            return
        versions = discovered.get("supportedVersions", [])
        if versions and MODERN_PROTOCOL_VERSION not in versions:
            await self._initialize_legacy()
            return
        self.server_info = dict(discovered.get("serverInfo", {}))
        self.server_capabilities = dict(discovered.get("capabilities", {}))

    async def _initialize_legacy(self) -> None:
        self.protocol_version = None
        result = await self.request("initialize", {
            "protocolVersion": LEGACY_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": self.client_info,
        })
        self.protocol_version = str(result.get("protocolVersion", LEGACY_PROTOCOL_VERSION))
        self.server_info = dict(result.get("serverInfo", {}))
        self.server_capabilities = dict(result.get("capabilities", {}))
        await self._write({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

    async def _write(self, payload: Mapping[str, Any]) -> None:
        if self.process is None or self.process.stdin is None:
            raise MCPError("MCP server stdin is unavailable")
        encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"
        self.process.stdin.write(encoded)
        try:
            await self.process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as error:
            raise MCPError(self._exit_message("MCP server closed stdin")) from error

    async def _read_response(self, request_id: int) -> dict[str, Any]:
        if self.process is None or self.process.stdout is None:
            raise MCPError("MCP server stdout is unavailable")
        while True:
            try:
                line = await asyncio.wait_for(self.process.stdout.readline(), timeout=self.timeout)
            except asyncio.TimeoutError as error:
                raise MCPError(f"MCP request {request_id} timed out after {self.timeout:g}s") from error
            if not line:
                raise MCPError(self._exit_message("MCP server closed stdout"))
            try:
                message = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise MCPProtocolError("MCP server wrote invalid JSON to stdout") from error
            if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
                raise MCPProtocolError("MCP server wrote an invalid JSON-RPC message")
            if message.get("id") == request_id:
                return message
            # Notifications have no id and can safely be ignored by this client.
            if "id" in message:
                raise MCPProtocolError(f"unexpected MCP response id: {message.get('id')!r}")

    async def _read_stderr(self) -> None:
        if self.process is None or self.process.stderr is None:
            return
        while line := await self.process.stderr.readline():
            self.stderr.append(line.decode("utf-8", errors="replace").rstrip())
            del self.stderr[:-100]

    def _exit_message(self, prefix: str) -> str:
        code = self.process.returncode if self.process is not None else None
        detail = f" (exit code {code})" if code is not None else ""
        if self.stderr:
            detail += f": {self.stderr[-1]}"
        return prefix + detail
