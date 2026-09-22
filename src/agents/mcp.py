"""First-class MCP execution contract using the existing secure MCP client."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from mcp.client import MCPClient, MCPProtocolError, MCPTool
from security.authorization import ToolAuthorization


@dataclass(frozen=True)
class MCPServerConfig:
    server_label: str
    server_url: str | None = None
    command: str | None = None
    args: tuple[str, ...] = ()
    approval_required: bool = True
    allowed_tools: frozenset[str] = frozenset()
    timeout_seconds: float = 30.0


class RemoteMCPClient:
    """Minimal Streamable-HTTP JSON-RPC MCP transport with bounded timeouts."""
    def __init__(self, url: str, *, timeout: float = 30.0, server_label: str = "mcp"):
        if not url.startswith(("https://", "http://")): raise ValueError("server_url must be http(s)")
        self.url, self.timeout, self.server_label = url, timeout, server_label
        self.client: httpx.AsyncClient | None = None
        self.request_id = 0
        self.protocol_version = "2025-06-18"
    async def start(self) -> None:
        if self.timeout <= 0:
            raise ValueError("MCP timeout must be positive")
        if self.client is not None:
            return
        self.client = httpx.AsyncClient(timeout=self.timeout, follow_redirects=False)
        try:
            await self.request("initialize", {"protocolVersion": self.protocol_version, "capabilities": {}, "clientInfo": {"name": "gopi-llm", "version": "1.0"}})
            await self.notify("notifications/initialized", {})
        except BaseException:
            await self.close()
            raise
    async def close(self) -> None:
        if self.client: await self.client.aclose(); self.client = None
    async def notify(self, method: str, params: Mapping[str, Any] | None = None) -> None:
        if self.client is None: raise RuntimeError("MCP client is not started")
        response = await self.client.post(self.url, json={"jsonrpc":"2.0","method":method,"params":dict(params or {})}, headers={"Accept":"application/json, text/event-stream"})
        response.raise_for_status()
    async def request(self, method: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if self.client is None: raise RuntimeError("MCP client is not started")
        self.request_id += 1
        payload={"jsonrpc":"2.0","id":self.request_id,"method":method,"params":dict(params or {})}
        response=await self.client.post(self.url,json=payload,headers={"Accept":"application/json, text/event-stream"})
        response.raise_for_status()
        data=response.json()
        if not isinstance(data, dict) or data.get("jsonrpc") != "2.0": raise MCPProtocolError("invalid remote MCP response")
        if isinstance(data.get("error"), dict): raise MCPProtocolError(str(data["error"].get("message","MCP request failed")))
        result=data.get("result")
        if not isinstance(result,dict): raise MCPProtocolError("remote MCP result is not an object")
        return result
    async def list_tools(self) -> list[MCPTool]:
        result=await self.request("tools/list", {})
        tools=result.get("tools", [])
        return [MCPTool.from_payload(item) for item in tools if isinstance(item, Mapping)]
    async def call_tool(self,name:str,arguments:Mapping[str,Any]|None=None,input_schema:Mapping[str,Any]|None=None)->dict[str,Any]:
        if input_schema is not None:
            from inference.local_tools import validate_json_schema
            validate_json_schema(dict(arguments or {}),input_schema)
        result=await self.request("tools/call",{"name":name,"arguments":dict(arguments or {})})
        if not isinstance(result.get("content",[]),list): raise MCPProtocolError("remote MCP tool result has invalid content")
        return result

class MCPExecutionContract:
    def __init__(self, config: MCPServerConfig):
        if not config.server_url and not config.command: raise ValueError("MCP server requires server_url or command")
        self.config = config
        self.authorization = ToolAuthorization(config.allowed_tools, config.approval_required, config.timeout_seconds)
        self.client: MCPClient | RemoteMCPClient | None = None
    async def start(self) -> None:
        if self.config.server_url:
            self.client = RemoteMCPClient(self.config.server_url, timeout=self.config.timeout_seconds, server_label=self.config.server_label)
        else:
            self.client = MCPClient([self.config.command, *self.config.args], timeout=self.config.timeout_seconds)
        try:
            await self.client.start()
            tools = await self.client.list_tools()
            available = {tool.name for tool in tools}
            unknown = self.config.allowed_tools - available
            if unknown:
                raise ValueError(f"configured MCP tools are unavailable: {', '.join(sorted(unknown))}")
        except BaseException:
            await self.client.close()
            self.client = None
            raise
    async def call(self, name: str, arguments: Mapping[str, Any] | None = None, *, approved: bool = False, input_schema: Mapping[str, Any] | None = None) -> dict[str, Any]:
        self.authorization.authorize(name, approved=approved)
        if self.client is None: raise RuntimeError("MCP contract is not started")
        return await self.client.call_tool(name, arguments, input_schema=input_schema)
    async def close(self) -> None:
        if self.client: await self.client.close(); self.client = None
