# Task Registry

This registry records active and planned repository tasks using stable IDs and
explicit dependencies.

### CORE-MODEL-001: Fix core-model review findings

- **ID**: `CORE-MODEL-001`
- **Status**: Complete
- **Dependencies**: None

Fixed the eight review findings covering cached training with checkpointing,
paged sliding-window attention, YaRN scaling, top-1 MoE gradients, causal MoE
capacity handling, strict MTP checkpoint loading, linear-state memory estimates,
and frozen heads during vocabulary resizing. Added 28 regression cases; all 163
targeted checks, including vocabulary compatibility, pass. Broader validation:
1,186 passed, 4 skipped, and 1 deselected with local socket access for distributed
CPU tests. The excluded dataset-governance test requires manifests already
deleted in the workspace; those unrelated deletions remain unchanged. Ruff lint,
format checks, and the task-registry audit pass.

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
## Batch 21 — next 20 missing-feature implementation

- **Status:** In progress; item 10 (recursive/dynamic JSON Schema references) completed and removed from the active audit.
- **Instruction:** `docs/BATCH21_IMPLEMENTATION_INSTRUCTIONS.md`
- **Scope:** audit items 1–20 from the post-Batch-20 missing-feature audit.
- **Rule:** remove an audit item only after source implementation, contract tests, regression validation, and documented runtime limitations are satisfied.
