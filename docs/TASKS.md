# Task Registry

This registry records active and planned repository tasks using stable IDs and
explicit dependencies.

### PRODUCTION-FEATURES-001: Implement and qualify production feature backlog

- **ID**: `PRODUCTION-FEATURES-001`
- **Status**: Complete
- **Dependencies**: None

The nineteen-item production feature backlog is tracked in
`docs/PRODUCTION_FEATURE_CHECKLIST.md`. Code completion and target-runtime
qualification are recorded separately so optional CUDA and vendor dependencies
cannot be mistaken for production evidence.

### PRODUCTION-PROFILE-001: Make production-profile qualification executable

- **ID**: `PRODUCTION-PROFILE-001`
- **Status**: Complete
- **Dependencies**: None

The profile-qualification CLI now executes reliably when invoked directly from
the repository and records evidence for the 100B sparse-MoE, hybrid-MoE 7B,
and 1T sparse-MoE profiles. A profile remains unqualified until its required
target CUDA and multi-node hardware evidence is observed.

### TRAINING-LOOP-001: Add an optimizer-step training limit

- **ID**: `TRAINING-LOOP-001`
- **Status**: Complete
- **Dependencies**: None

Language-model training now accepts an optional absolute `max_steps` limit from
the training YAML or `--max-steps`. It stops only after a completed optimizer
update, writes the standard resumable checkpoint with its in-epoch batch
position, and uses the smaller epoch/step duration for a fresh scheduler plan.

### TRAINING-LOOP-002: Add bounded-run and recovery controls

- **ID**: `TRAINING-LOOP-002`
- **Status**: Complete
- **Dependencies**: TRAINING-LOOP-001

The language-model trainer now supports global micro-batch and supervised-token
limits, wall-clock checkpoint intervals, and optional automatic resume from the
configured latest checkpoint. All termination paths save a resumable position.

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

### INFERENCE-002: Repair serving adapter and paged-KV lifecycle safety

- **ID**: `INFERENCE-002`
- **Status**: Complete
- **Dependencies**: INFERENCE-001

Token-step streams snapshot their tenant LoRA state at admission and batch only
streams with that same snapshot. Distinct adapters are decoded in separate,
lock-protected calls, so a mutable adapter cannot cross tenant boundaries while
compatible streams still batch together. Paged-KV admission is transactional,
allocator memory reporting reflects actual reserved storage, and
`serving.paged_kv_quantization` exposes the existing `none`/`int8` cache
formats. The INT8 option requires target-model quality and throughput
qualification before production use.

The active 4 GB serving profile also reserves enough pages for four 1,024-token
streams, bounds prefill to 256 tokens per pass, and caps retained prefixes to
four full-context entries. MTP auxiliary heads intentionally remain a
training-only objective: with their current horizon-2+ contract, treating them
as an inference draft would require an additional target pass and offers no
safe speedup.

When adding a task, use a level-three heading with a stable identifier, a
matching `ID` field, and a `Dependencies` field. The registry audit validates
these fields and verifies that every listed dependency exists.
