# Task Registry

This registry records active and planned repository tasks using stable IDs and
explicit dependencies.

### QUALITY-001: Add a pull-request regression gate

- **ID**: `QUALITY-001`
- **Status**: Complete
- **Dependencies**: None

GitHub Actions now installs the project with its development dependencies, runs
the complete pytest suite, and validates this task registry for pushes to
`main` and pull requests.

### INFERENCE-001: Add bounded long-context prompt prefill

- **ID**: `INFERENCE-001`
- **Status**: Complete
- **Dependencies**: None

The optional `serving.prefill_chunk_size` control evaluates a long prompt in
bounded chunks while retaining an allocation-efficient fixed-capacity KV cache.
The default remains one-pass prefill for unchanged behavior.

When adding a task, use a level-three heading with a stable identifier, a
matching `ID` field, and a `Dependencies` field. The registry audit validates
these fields and verifies that every listed dependency exists.
