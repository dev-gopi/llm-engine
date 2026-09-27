# llm-engine current missing-feature audit — post batch review

Audit basis: `llm-engine-main(12)-online-grpo-reward-calibration.zip` plus the implementation changes in this batch.

The original audit intentionally contained only missing/incomplete features. After this batch, the completed items from the first 20 have been removed below. Features that still require production hardware validation or are not fully implemented remain listed.

## Remaining model/post-training gaps

1. **100B sparse-MoE / hybrid-MoE 7B / 1T production execution profiles** remain incomplete. Meta-device construction, expert-shard helpers, and parallel topology contracts exist, but a real multi-node training/checkpoint/resume qualification for these scales has not been performed in this CPU audit environment.

## Remaining inference/decoding gaps

2. Native CUDA low-bit inference beyond CPU dynamic INT8/custom export paths.
3. True native GPTQ conversion/runtime.
4. True native AWQ conversion/runtime.
5. True GGUF binary conversion in the main export CLI.
6. Hot-swappable LoRA adapter serving with lifecycle/version/tenant isolation.
7. Native vLLM backend integration.
8. Native TensorRT-LLM backend integration.
9. Native FlashInfer backend integration.
10. Production serving-integrated speculative decoding with draft-model lifecycle.
11. Native FlashAttention-2/FlashAttention-3 package backend.
12. Grammar-constrained decoding beyond the current JSON-schema path.

## Remaining multimodal gaps

13. Production native latent-image profile; the latent production profile remains intentionally planning-only until production training/inference validation is completed.
14. Concrete general-purpose audio-understanding provider/runtime.
15. Voice-cloning provider/training/runtime.
16. Standalone pretrained video-understanding backend.
17. Broader interchangeable vendor/provider adapters for image/audio/video generation/editing/understanding.
18. Full audio-only Responses modality parity where text is not required.

## Remaining semantic-cache production gaps

19. Distributed Redis/vector-index semantic-cache backend.
20. Cross-replica single-flight coordination.
21. Multi-tenant/user cache namespaces with quotas.
22. Per-request `no-store` / privacy control.
23. Automatic sensitive-data cache exclusion policy/classifier.
24. RAG/tool/web/session freshness fingerprints; these requests are currently safely bypassed.
25. Negative/error caching with separate short TTL policy.
26. Cache warming/prepopulation.
27. Selective purge by model/tenant/time/key rather than namespace-wide clear only.
28. Token/cost/latency-saved and similarity-distribution metrics.
29. Explain/debug metadata for semantic-match selection.
30. Offline false-positive / answer-drift semantic-cache evaluation tooling.
31. Stale-while-revalidate/background refresh.
32. Per-route/model/task similarity thresholds.

## Remaining API/platform gaps

33. OAuth/OIDC authentication.
34. SSO and tenant-aware RBAC.
35. Tenant data isolation/quotas as a first-class platform layer.
36. Built-in billing/metering product layer beyond usage-accounting primitives.
37. Full content-moderation service/model pipeline.
38. Turnkey compliance controls: data residency, KMS/HSM, legal holds, enterprise retention/compliance reporting.
39. Kubernetes/operator/autoscaler deployment layer.
40. Turnkey multi-node serving with replica routing/autoscaling.
41. OpenTelemetry runtime/exporter despite the configuration flag.
42. Native OpenAI-style files/vector-stores/batches/fine-tuning-jobs product APIs.
43. Native `/v1/images/generations` compatibility endpoint; media generation uses project-specific routes.
44. Additional MCP transports beyond the implemented transport set.
45. Broader legacy-tool schema parity; richer tools exist through the newer tool path.
46. First-class request idempotency keys for mutation/job endpoints and replay-safe distributed handling.
47. Signed webhook/event delivery for async jobs with retry and dead-letter handling.

## New follow-up gaps discovered during this batch audit

48. GPU qualification of FSDP RLHF sharded save/resume across real multi-rank CUDA jobs.
49. GPU/multi-node qualification of context/sequence/pipeline/expert parallel collectives and MoE tensor-parallel serving.
50. Production container/VM isolation qualification for the Python sandbox verifier; the code-level verifier has timeout/process isolation but OS sandboxing must be supplied by deployment.
51. Production load/latency qualification of the reward HTTP service and its batch endpoint.
52. End-to-end integration of optional MLflow/W&B/TensorBoard adapters into every trainer entry point; adapters are implemented, but trainer wiring remains opt-in.

## Validation status for this iteration

- New batch contract tests: **9 passed**.
- Existing GRPO/reward/distributed/MoE/parallel focused tests plus new batch: **61 passed**.
- Broader regression selection covering training/config/model/inference/serving/DPO: **168 passed, 2 skipped**.
- `python -m compileall -q src scripts`: **PASS**.
- No existing focused regression failure was introduced by this batch.
- Full repository collection was not claimed as a pass; the original audit already documented the `pyarrow` limitation and timeout on the full collection.
