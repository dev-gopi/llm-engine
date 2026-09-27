# Batch 23 — implementation instructions

## Root and baseline

Use this Batch-23 repository as the root. Do not rewrite or remove existing working code from Batch 22. Every change must be additive, backwards-compatible, and covered by focused regression tests.

## Scope

Implement the next 20 unresolved capabilities from the Batch-22 audit:

1. 100B sparse-MoE / hybrid-MoE 7B / 1T production execution profiles
2. Native CUDA low-bit inference
3. Native GPTQ conversion/runtime
4. Native AWQ conversion/runtime
5. Native GGUF binary export
6. Hot-swappable LoRA adapter serving with lifecycle/version/tenant isolation
7. vLLM backend
8. TensorRT-LLM backend
9. FlashInfer backend
10. Production speculative decoding with draft-model lifecycle
11. FlashAttention-2/3 backend
12. Grammar-constrained decoding beyond JSON Schema
13. Production latent-image profile
14. General-purpose audio-understanding provider/runtime
15. Voice-cloning provider/training/runtime
16. Standalone pretrained video-understanding backend
17. Interchangeable multimodal provider adapters
18. Audio-only Responses parity
19. Live Redis semantic-cache qualification
20. ServingRuntime distributed semantic-cache integration

## Accuracy rules

- Never claim a vendor/runtime feature is production-complete unless its runtime is installed and an end-to-end test exercises it.
- Optional dependencies must be lazy-loaded and must fail explicitly when unavailable.
- Never silently fall back from a requested backend or format to another backend/format.
- Preserve all existing public APIs unless an additive compatibility layer is required.
- Do not label the engine's research INT4/Q1 representation as GPTQ, AWQ, or GGUF.
- Hardware-dependent qualification must be recorded separately from code-level implementation.
- Security-sensitive multimodal features require explicit policy enforcement.
- Run focused regression tests after every subsystem, then the broader regression selection.
- Run `python -m compileall -q src scripts` before packaging.

## Acceptance

A feature can be removed from the missing-feature audit only when its implementation and required qualification evidence are present. If the environment lacks CUDA, vendor packages, Redis, or multi-node infrastructure, keep the qualification item open and document the exact limitation.
