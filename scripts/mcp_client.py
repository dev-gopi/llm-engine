"""List and invoke tools on a configured MCP stdio server."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

from mcp.client import MCPClient, MCPError
from utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "call"))
    parser.add_argument("tool", nargs="?")
    parser.add_argument("arguments", nargs="?", default="{}", help="JSON object for a tool call")
    parser.add_argument("--config", type=Path, default=Path("configs/mcp.yaml"))
    parser.add_argument("--server", required=True, help="Server key under mcp.servers")
    return parser.parse_intermixed_args()


async def run(args: argparse.Namespace) -> None:
    root = load_yaml(args.config)
    config = root.get("mcp", {})
    if not isinstance(config, dict):
        raise ValueError("mcp configuration must be a mapping")
    servers = config.get("servers", {})
    if not isinstance(servers, dict):
        raise ValueError("mcp.servers must be a mapping")
    if args.server not in servers:
        raise ValueError(f"MCP server is not configured: {args.server}")
    server = servers[args.server]
    if not isinstance(server, dict):
        raise ValueError(f"MCP server configuration must be a mapping: {args.server}")
    command = [str(server["command"]), *(str(value) for value in server.get("args", []))]
    async with MCPClient(
        command,
        cwd=server.get("cwd"),
        env=server.get("env"),
        timeout=float(server.get("timeout_seconds", 30)),
        protocol=str(server.get("protocol", "auto")),
        max_message_bytes=int(server.get("max_message_bytes", 16 * 1024 * 1024)),
    ) as client:
        if args.action == "list":
            result = [tool.__dict__ for tool in await client.list_tools()]
        else:
            if not args.tool:
                raise ValueError("call requires a tool name")
            arguments = json.loads(args.arguments)
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must be a JSON object")
            result = await client.call_tool(args.tool, arguments)
        print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run(args))
    except (FileNotFoundError, KeyError, MCPError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
