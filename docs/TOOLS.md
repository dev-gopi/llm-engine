# Tool Calling & Execution Engine (`docs/TOOLS.md`)

*Authoritative Source: [`src/inference/local_tools.py`](../src/inference/local_tools.py), [`src/mcp/client.py`](../src/mcp/client.py), [`configs/mcp.yaml`](../configs/mcp.yaml)*

---

## 1. Tool Declaration & Schema Standard

Tools are defined using standard JSON Schema definitions compatible with OpenAI and MCP standards.

### Example Tool Definition:
```json
{
  "name": "calculate",
  "description": "Evaluate an arithmetic expression safely",
  "parameters": {
    "type": "object",
    "properties": {
      "expression": {
        "type": "string",
        "description": "Mathematical expression, e.g. '25 * 37 + 12'"
      }
    },
    "required": ["expression"]
  }
}
```

---

## 2. Built-in Local Tools

Configured in [`src/inference/local_tools.py`](../src/inference/local_tools.py):

| Tool Name | Parameters | Safety Constraints |
| :--- | :--- | :--- |
| `read_file` | `path: str` | Path MUST reside inside the allowed workspace directory |
| `list_directory`| `path: str` | Path MUST reside inside workspace directory |
| `search_files` | `pattern: str, path: str` | Bounded file search with maximum match limit |
| `calculator` | `expression: str` | AST-parsed arithmetic; no arbitrary code execution (`eval`) |
| `web_search` | `query: str` | Routed through SearXNG privacy search proxy |

---

## 3. Tool Execution Protocol

```text
1. Prompt Construction:
   System prompt lists active tool definitions in JSON format.

2. Model Emission:
   Model outputs structured delimiter block:
   <tool_call>
   {"name": "calculator", "arguments": {"expression": "144 ** 0.5"}}
   </tool_call>

3. Schema Validation:
   Arguments parsed and checked against JSON Schema types. Malformed JSON triggers
   an automated error feedback message to the model without crashing the agent.

4. Execution:
   The tool executor runs the function and captures output or exceptions.

5. Result Injection:
   Result formatted as:
   <tool_result>
   {"result": 12.0}
   </tool_result>
   Fed back to model for final generation.
```

