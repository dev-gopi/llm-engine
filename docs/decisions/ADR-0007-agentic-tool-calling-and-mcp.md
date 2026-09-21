# ADR-0007: Tool Calling via Model Context Protocol (MCP) and JSON Schemas

## Status
`ACCEPTED`

## Context
Autonomous agents require interaction with tools (calculators, filesystems, search engines, databases). Hardcoding proprietary tool bridges leads to brittle implementations and high integration maintenance.

## Decision
We adopted the **Model Context Protocol (MCP)** standard for external tool orchestration, complemented by sandboxed built-in local workspace tools and strict JSON schema argument parsing.

## Alternatives Considered
- **Custom Ad-Hoc Python Function Callers**: Fragile, non-standardized schema exchange.
- **LangChain / LlamaIndex Integration**: Heavy external dependencies with frequent breaking API changes and high context token overhead.

## Consequences
- **Positive**: Interoperable with any standard MCP tool server; strict schema validation; isolated subprocess execution.
- **Negative**: Adds async JSON-RPC message passing overhead.

## Related Files
- [`src/mcp/client.py`](../../src/mcp/client.py)
- [`src/inference/local_tools.py`](../../src/inference/local_tools.py)
- [`configs/mcp.yaml`](../../configs/mcp.yaml)

