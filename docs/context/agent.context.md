# Compact Context: Agent & Tools (`docs/context/agent.context.md`)

> **AGENT CONTEXT PACK**: Load this file when working on agent loops, function calling, tool execution, MCP integration, or RAG.

---

## 1. Authoritative Sources of Truth
- **Agent Orchestrator**: [`src/serving/workspace.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/serving/workspace.py)
- **MCP Client**: [`src/mcp/client.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/mcp/client.py)
- **Local Tools**: [`src/inference/local_tools.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/inference/local_tools.py)
- **RAG Engine**: [`src/inference/rag.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/inference/rag.py)
- **Tool Config**: [`configs/mcp.yaml`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/configs/mcp.yaml)

---

## 2. Tool Execution Workflow
1. System prompt defines available tool JSON schemas.
2. Model emits `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`.
3. Schema validator parses and checks types against JSON schema.
4. Executor runs tool (local function or MCP server) within allowed directory sandbox.
5. Injects `<tool_result>{...}</tool_result>` into conversation for second-pass generation.

---

## 3. Key Invariants
1. All file paths must be validated against `workspace_dir` using `os.path.realpath`.
2. No unrestricted shell commands allowed.
3. Tool execution failures must be caught and returned as structured error strings, never raising unhandled exceptions in the agent loop.

---

## 4. Primary Verification Tests
```bash
.venv/bin/pytest tests/test_local_tools.py tests/test_mcp_client.py tests/test_workspace_agent.py tests/test_rag.py -q
```

