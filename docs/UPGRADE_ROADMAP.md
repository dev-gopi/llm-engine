# Upgrade Roadmap (`docs/UPGRADE_ROADMAP.md`)

This file is a **derived planning view**. The authoritative task IDs, status,
dependencies, acceptance criteria, and validation commands live in
[`TASKS.md`](TASKS.md). Do not create or redefine task IDs here.

*Reconciled: 2026-09-21*

## Current registry state

- **97** unique task IDs.
- **81** tasks are source/test-verified as `COMPLETED`.
- **16** tasks remain `TODO` because either implementation is incomplete or the
  acceptance criteria require external datasets, checkpoints, GPU measurements,
  long-duration runs, or production evidence that is not present in this ZIP.
- Run `python scripts/audit_task_registry.py` to validate unique IDs and
  dependency references.

## Remaining work

| Task | Priority | Remaining acceptance work |
| --- | --- | --- |
| `DATA-003` | P0 | Acquire and review real multi-domain corpora; publish provenance, licensing, quality, and contamination evidence before activation. |
| `DATA-004` | P0 | Run the contamination audit against the actual governed train/evaluation corpora. Unit coverage alone is not production evidence. |
| `CTX-002` | P0 | Execute checkpoint-backed 2K/4K/8K retrieval and CUDA peak-memory measurements on matching trained checkpoints. |
| `ALIGN-001` | P0 | Complete preference-pair provenance and bias auditing beyond the existing quality/duplicate/secret checks. |
| `SPC-001` | P3 | Run a real draft/target checkpoint throughput comparison and record acceptance/latency evidence. |
| `VIS-002` | P3 | Govern a real image-text corpus and run checkpoint-backed multimodal instruction evaluation. |
| `QNT-002` | P2 | Implement and verify real GPTQ/AWQ/GGUF-compatible artifact conversion/import, then record quality, latency, and memory regressions. |
| `ATT-002` | P3 | Benchmark advanced attention profiles against dense GQA and validate checkpoint compatibility. |
| `SCALE-004` | P2 | Integrate a calibrated low-bit base-model training path for QLoRA and verify merge/export quality. |
| `SCALE-005` | P3 | Validate actual tensor/pipeline/expert-parallel execution and checkpoint behavior, not only topology contracts. |
| `VIS-003` | P3 | Add versioned audio/video datasets, encoders/projectors, safety controls, and evaluation evidence. |
| `AGT-005` | P1 | Add versioned deterministic end-to-end agent fixtures; the current metric aggregator alone is not sufficient. |
| `CTX-003` | P1 | Record the requested 512→1K→2K→4K→8K quality/memory/throughput ablation with real runs. |
| `INF-003` | P1 | Run end-to-end serving benchmarks across batch sizes and precisions and retain TTFT/ITL/TPS/memory artifacts. |
| `INF-004` | P1 | Execute sustained 10m/1h/6h/24h load/soak runs and retain leak/error evidence. |
| `INF-006` | P1 | Extend failure stress from contracts/simulations to real disconnect, timeout, OOM, tool-failure, and restart cleanup runs. |

## Recently reconciled as completed

The 2026-09-21 audit found implementation and regression coverage that the
registry still labeled `TODO`. The authoritative registry now marks these as
`COMPLETED`: `FTL-001`, `AGT-004`, `MEM-002`, `TRAIN-005`, `CHAT-003`,
`ALIGN-002`, `ALIGN-003`, `RSN-004`, `RSN-005`, `EVAL-003`, `EVAL-004`,
`EVAL-005`, `EVAL-006`, `EVAL-007`, `TOKEN-002`, `TOKEN-003`, and `RSN-007`.
See [`TASKS.md`](TASKS.md) for the exact files, tests, and acceptance criteria.

## Execution order

A practical dependency-first sequence is:

1. **Data evidence:** `DATA-003` → `DATA-004` → `ALIGN-001`.
2. **Checkpoint-backed capability evidence:** `CTX-002` → `CTX-003`, plus
   `VIS-002` and `SPC-001` when matching checkpoints are available.
3. **Deployment formats and scale:** `QNT-002` → `SCALE-004`; independently
   validate `SCALE-005` and `ATT-002`.
4. **Production performance:** `INF-003` → `INF-004` → `INF-006`.
5. **Agent and multimodal research:** finish `AGT-005` and `VIS-003` using
   governed fixtures and release-gated evaluation artifacts.

No task should be changed to `COMPLETED` solely because a helper class or unit
test exists when its own acceptance criteria explicitly require external run
evidence.
