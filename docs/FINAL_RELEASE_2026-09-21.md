# Final Enhanced Release — 2026-09-21

This release is based on the last audited codebase and preserves the default model/checkpoint path while adding opt-in modern model/runtime features.

## Implemented enhancement set

- Hybrid linear+dense attention scheduling with fixed-state causal linear decoding.
- QK normalization support.
- Sparse MoE routing diagnostics and auxiliary load-balancing loss.
- MTP training/report integration.
- Q1_0-style 1.125-bit research packing/export for the engine-native format.
- Deployment memory/resource planning for BF16, INT8, INT4 and Q1 research precision.
- External OpenAI-compatible inference backend and GGUF/llama.cpp serving launcher.
- Expanded `/v1/models` architecture/capability metadata.
- Reasoning-effort, usage, architecture and resource visibility in the playground.
- Expanded training report for architecture, deployment footprint, MTP and MoE metrics.
- Modern hybrid 8K model profile and planning-only larger hybrid/MoE profiles.
- Updated task registry, current-state docs, changelog and research notes.

## Final release integrity

- Task IDs: 104 unique.
- Statuses: 88 COMPLETED, 16 TODO.
- Duplicate task IDs: 0.
- Final enhancement smoke gate: 213 passed.

Command used for the final smoke gate:

```bash
PYTHONPATH=src python -m pytest \
  tests/test_attention.py \
  tests/test_attention_research.py \
  tests/test_model_config.py \
  tests/test_loss.py \
  tests/test_training_system.py \
  tests/test_serving.py \
  tests/test_external_backend.py \
  tests/test_inference_precision.py \
  tests/test_training_report_builder.py \
  -q
```

## Important validation boundary

The repository implements the code paths and interfaces above, but tasks that explicitly require external corpora, trained large checkpoints, long-duration soak tests, or target-GPU measurements remain open until that evidence exists. GGUF models such as Bonsai-27B-Q1_0 are delegated to a compatible optimized runtime rather than falsely emulating architecture-specific low-bit kernels in the native Python model path.
