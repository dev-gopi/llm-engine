# Batch 21 Implementation Instructions

## Objective
Implement the next 20 items from `llm-engine-main-batch20-missing-features-audit-updated.md` without breaking existing behavior. The audit is the source of truth for scope.

## Root
The working repository root is the extracted `llm-engine-main` directory. Do not modify files outside this repository.

## Scope
Implement audit items 1–20 only:

1. 100B sparse-MoE / hybrid-MoE 7B / 1T production execution profiles.
2. Native CUDA low-bit inference beyond CPU dynamic INT8/custom export paths.
3. Native GPTQ conversion/runtime.
4. Native AWQ conversion/runtime.
5. Native GGUF binary conversion in the main export CLI.
6. Hot-swappable LoRA adapter serving with lifecycle/version/tenant isolation.
7. Native vLLM backend integration.
8. Native TensorRT-LLM backend integration.
9. Native FlashInfer backend integration.
10. Recursive/dynamic JSON Schema references in structured-output constraints.
11. Production serving-integrated speculative decoding with draft-model lifecycle.
12. Native FlashAttention-2/FlashAttention-3 package backend.
13. Grammar-constrained decoding beyond the current JSON-schema path.
14. Production native latent-image profile.
15. General-purpose audio-understanding provider/runtime.
16. Voice-cloning provider/training/runtime.
17. Standalone pretrained video-understanding backend.
18. Interchangeable vendor/provider adapters for image/audio/video generation/editing/understanding.
19. Full audio-only Responses modality parity where text is not required.
20. Distributed Redis/vector-index semantic-cache backend.

## Non-negotiable safety rules
- Preserve all existing public APIs unless backward-compatible extension is required.
- Never replace a working implementation merely to rename it.
- No hardcoded production secrets, credentials, endpoints, tenant IDs, or cloud resources.
- Hardware-specific features must have CPU-safe contract tests and must fail clearly when their optional runtime is unavailable.
- Optional dependencies must remain optional; importing the base package must not require CUDA/vLLM/TensorRT/FlashInfer/FlashAttention/Redis/etc.
- Do not mark a feature production-ready solely from a mock or CPU test. Record qualification requirements explicitly.
- Configuration belongs in `configs/*.yaml`, not magic constants in source.
- Preserve checkpoint compatibility and run vocabulary compatibility tests after model changes.

## Required workflow per feature
1. Locate authoritative files through `docs/PROJECT_INDEX.md`.
2. Inspect existing implementation and tests before editing.
3. Add the smallest compatible implementation.
4. Add unit/contract tests for behavior and unavailable-runtime paths.
5. Run targeted tests.
6. Run the relevant regression selection.
7. Run `python -m compileall -q src scripts`.
8. Update `docs/TASKS.md`, `docs/CHANGELOG.md`, and an ADR when architectural behavior changes.
9. Re-audit the missing-feature sheet and remove an item only when its acceptance criteria are actually met.

## Acceptance standard
An item may be marked **implemented** only when:
- the source implementation exists;
- its public/configuration contract is documented;
- normal and failure paths are tested;
- existing relevant tests pass;
- optional runtime dependencies are guarded;
- production/hardware qualification requirements are explicitly separated from code-level completion.

## Batch output
Produce:
- patched source tree;
- targeted regression tests;
- updated project documentation;
- updated missing-feature audit with completed items removed and unresolved qualification gaps retained;
- concise implementation report listing files changed, tests run, and known limitations;
- reproducible ZIP archive of the patched repository.
