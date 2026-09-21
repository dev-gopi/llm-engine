# Security & Safety Guardrails (`docs/SECURITY.md`)

*Authoritative Source: [`src/inference/prompt_safety.py`](../src/inference/prompt_safety.py), [`src/serving/rate_limit.py`](../src/serving/rate_limit.py), [`src/inference/local_tools.py`](../src/inference/local_tools.py)*

---

## 1. Prompt Injection Defenses & Input Sanitization

[`src/inference/prompt_safety.py`](../src/inference/prompt_safety.py) provides heuristic and pattern-based protection against jailbreak and prompt-injection attacks:
- **Role Token Sanitization**: Verifies that user-supplied input does not forge internal delimiters such as `<|system|>`, `<|assistant|>`, or `<|tool_result|>`. Any injected control tags are escaped before tokenization.
- **Untrusted Source Delimiters**: External web search and RAG document results are wrapped in explicit untrusted tags:
  ```text
  <untrusted_context source="doc_1">
  ... content ...
  </untrusted_context>
  ```
- **Prompt Safety Classifiers**: Scans incoming prompts against forbidden instruction patterns and jailbreak templates.

---

## 2. Tool Execution Sandbox & Path Confinement

Local tools in [`src/inference/local_tools.py`](../src/inference/local_tools.py) enforce strict security boundaries:
- **Strict Directory Jail**: All file read/write operations resolve canonical realpaths (`os.path.realpath`) and verify that the target begins with the allowed workspace root (`allowed_dir`).
- **Path Traversal Blocking**: Patterns containing `..`, absolute paths outside workspace, or symbolic link escapes are rejected immediately with permission errors.
- **No Arbitrary Shell Execution**: The agent never invokes an unrestricted shell (`/bin/sh` or `bash -c`). Tools execute strictly allowlisted Python functions with type-validated arguments.

---

## 3. Serving Security & Rate Limiting

- **Token Bucket Rate Limiting**: Implemented in [`src/serving/rate_limit.py`](../src/serving/rate_limit.py) with SQLite backing (`data/cache/rate-limits.sqlite`). Restricts burst and sustained request rates per IP or client token.
- **TLS / SSL Termination**: Production deployments route traffic through Nginx with TLS 1.3 certificates (`deploy/certs/`).
- **CORS Configuration**: Restricts WebSocket and HTTP origins to authorized domain origins.

