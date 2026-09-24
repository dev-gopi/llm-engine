# Task Registry

This registry records active and planned repository tasks using stable IDs and
explicit dependencies.

### TRAINING-VALIDATION-001: Enable validation-driven plateau recovery

- **ID**: `TRAINING-VALIDATION-001`
- **Status**: Complete
- **Dependencies**: None

The primary CPU and GPU training profiles now lower the scheduled learning rate
after a held-out validation plateau, while retaining the best checkpoint and
using early stopping as the final guardrail. Model architecture is deliberately
not changed during a run, preserving checkpoint compatibility.
The `validation_lr_adaptation_enabled` switch enables or disables only learning-rate adaptation; best-checkpoint selection and early stopping remain active in either mode.

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
