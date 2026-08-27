import json
import os
import sys


def respond(request, *, result=None, error=None):
    payload = {"jsonrpc": "2.0", "id": request["id"]}
    payload["error" if error else "result"] = error or result
    print(json.dumps(payload), flush=True)


for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    method = request["method"]
    if method == "server/discover":
        if os.getenv("FAKE_MCP_MODERN") == "1":
            respond(request, result={
                "supportedVersions": ["2026-07-28"],
                "serverInfo": {"name": "fake-modern", "version": "1"},
                "capabilities": {"tools": {}},
            })
        else:
            respond(request, error={"code": -32601, "message": "method not found"})
    elif method == "initialize":
        respond(request, result={
            "protocolVersion": "2025-06-18",
            "serverInfo": {"name": "fake-legacy", "version": "1"},
            "capabilities": {"tools": {}},
        })
    elif method == "tools/list":
        respond(request, result={"tools": [{
            "name": "echo", "description": "Echo text",
            "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}},
        }]})
    elif method == "tools/call":
        arguments = request.get("params", {}).get("arguments", {})
        respond(request, result={"content": [{"type": "text", "text": arguments.get("text", "")}], "isError": False})
    else:
        respond(request, error={"code": -32601, "message": "method not found"})
