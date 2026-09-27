## Batch 24 — PPO + distributed online GRPO
- Added end-to-end sequence-level PPO RLHF training with actor/reference/reward/value lifecycle, adaptive KL, rollout collection, resume, and separate value checkpoints.
- Added CPU/GPU PPO profiles and promoted PPO from planned to operational alignment methods.
- Wired online GRPO for DDP prompt sharding, cross-rank rollout aggregation, distributed sampling/training, rank-0 persistence, and distributed metadata.
- Added focused regression coverage and refreshed the current missing-feature audit.


## Batch 22 — semantic cache production layer
- Added optional Redis semantic-cache backend with distributed single-flight primitives.
- Added tenant quotas, privacy/no-store policy, sensitive-data exclusion, dynamic-request freshness fingerprints, negative caching, warming, selective purge, and route/model/task thresholds.
- Added cache metrics for token/latency savings and similarity distributions plus explainable selection metadata.
- Added offline semantic-cache false-positive/answer-drift evaluation tooling.
- Preserved the existing local SQLite semantic cache and public behavior.

## 2026-09-27 — FSDP post-training wiring

- Enabled FSDP/hybrid-FSDP execution in DPO/IPO/ORPO/KTO, reward-model, and offline-GRPO CLIs.
- Added v2 rank-sharded post-training checkpoint manifests with optimizer, scheduler, scaler, trainer state, and world-size validation.
- Preserved single-process/DDP checkpoint compatibility.
- Added FSDP wiring regression tests and documentation.

## 2026-09-27 — Distributed state + multimodal completion batch

- Added Redis Search HNSW semantic-cache indexing with plain-Redis fallback and atomic cross-replica semantic-cache quotas.
- Added Redis-backed request rate limiting, idempotency, session history and OpenAI-platform lifecycle metadata.
- Added a production native latent-image provider and latent-diffusion training entry point/profile.
- Added configurable Hugging Face audio-understanding and standalone video-understanding providers.
- Added consent-gated Coqui XTTS voice-cloning runtime.
- Added audio-only Responses output support.
- Added realtime multimodal WebSocket events and optional WebRTC data-channel transport.
