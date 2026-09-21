# Tool-Use SFT and Evaluation Contract

`AGT-003` uses `configs/evaluation.tool_use.jsonl` as a versioned, deterministic
fixture manifest. Each row defines an allowed-tool set, one or more calls, the
argument JSON schema, execution intent, and expected result or structured error.

Rows cover selection, schema rejection, result handling, error recovery,
sequential execution, parallel intent, and permission denial. `parallel_intent`
is training/evaluation data only: the current MCP runtime executes bounded calls
sequentially, so it must not be interpreted as authorization to run tools in
parallel. A call outside `allowed_tools` is a required denial.

External SFT examples must additionally pass the adjacent dataset manifest's
license, privacy, provenance, and reviewer checks before activation.
