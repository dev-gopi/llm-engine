# LLM Engine — Current Missing Feature Audit

Audit basis: source tree after main(16) platform-workers + enterprise completion batch.

## Closed at code level in main(16)

### Platform workers
- Vector-store parse/chunk/embed/index/search worker with durable SQLite/Redis execution state.
- Batch JSONL validation/execution/output/error worker with persisted request counts.
- Fine-tuning Jobs orchestration worker that generates an executable SFT config, launches training, captures logs/errors and registers the result.
- Shared worker discovery/state transitions for SQLite and Redis control planes.
- One-shot/daemon worker CLI suitable for CronJob or long-running worker deployment.

### Enterprise/control plane
- Billing/product entitlement and usage-metering source implementation.
- Moderation pipeline with local rules and injectable model classifier.
- Data-residency policy, legal hold, retention and compliance reporting.
- SCIM-style identity/group directory and local/external policy engine.
- Tamper-evident signed audit chain/export primitives.
- Model/artifact hashing, SBOM metadata, signing and admission verification.
- Environment/AWS Secrets Manager/Vault secret retrieval and rotation-compatible lookup path.
- Local/AWS KMS/optional PKCS#11 HSM signing adapter.
- Hardened sandbox wrapper using bubblewrap/firejail when installed.
- Kubernetes Deployment/HPA generation and SLO-based promotion/rollback decisions.
- Replica discovery/health routing, multi-region failover selection and backup/restore primitives.
- Multi-destination tenant webhook subscription registry.
- Project-owned runtime/source paths contain no literal `NotImplementedError` placeholders.

## Remaining model / distributed-training integration gaps
1. DeepSpeed/ZeRO is implemented as an adapter but is not yet the first-class strategy used by every trainer.
2. DP/TP/PP/EP/CP/SP process mesh is not yet selected uniformly from all trainer configs.
3. Context-parallel attention is not yet wired directly through MiniGPT attention forward.
4. Sequence-parallel execution is not yet wired through all transformer blocks.
5. Pipeline stage transport exists, but model partitioning + multi-microbatch 1F1B is not integrated into the main trainer.
6. Expert-parallel all-to-all exists, but SparseMoE forward/optimizer/checkpoint lifecycle is not fully integrated with trainer topology selection.
7. Elastic post-training FSDP resharding across changed world sizes remains incomplete.
8. Multi-node restart/fault qualification remains outstanding.
9. 7B/100B/1T MoE profiles remain hardware/data qualification targets.
10. Experiment tracking is not instantiated consistently by every trainer.
11. Full encoder-decoder/seq2seq train/export/serve lifecycle remains incomplete.

## Remaining inference integration / qualification gaps
12. TensorRT-LLM generation adapter requires real engine/version qualification.
13. FlashInfer execution is implemented but is not automatically selected by MiniGPT attention.
14. FlashAttention execution is implemented but is not automatically selected by MiniGPT attention.
15. torchao INT4/INT8/FP8 paths require supported-GPU qualification.
16. GPTQ/AWQ conversion paths require target-runtime/calibration qualification.
17. GGUF source export requires target llama.cpp quantization/architecture qualification.
18. Async vLLM exists but normal serving still defaults to the buffered synchronous adapter.
19. Circuit breaker/failover router is not yet the default backend orchestration path.
20. Tenant QoS queue is not yet the default serving admission scheduler.
21. Multi-node TP/PP/EP inference is not unified under one serving topology configuration.

## External qualification still required
Source-level implementation does not certify external infrastructure. Production qualification remains required for real Redis/Redis Stack multi-replica behavior, cloud KMS/secret managers, Vault, PKCS#11 HSMs, hardened sandbox tools, Kubernetes clusters/autoscalers, multi-region networking/storage/failover, production moderation models, real fine-tuning hardware, and vendor GPU inference backends.
