# Current implementation status

## Batch 26 completed
- [x] Server-configured Redis semantic cache
- [x] Distributed semantic nearest-neighbor lookup with policy isolation
- [x] Distributed streaming semantic cache
- [x] Tenant cache quota enforcement
- [x] Cache warming/selective purge/negative-cache admin operations
- [x] Hot-swappable LoRA lifecycle API
- [x] EBNF grammar constraints in generate + chat APIs
- [x] OIDC/JWT authentication
- [x] RBAC admin authorization
- [x] Authenticated tenant/user binding
- [x] Per-tenant request quotas
- [x] OpenTelemetry tracing / OTLP export
- [x] OpenAI-compatible Files API
- [x] OpenAI-compatible Vector Stores lifecycle API
- [x] OpenAI-compatible Batch lifecycle API
- [x] OpenAI-compatible Fine-tuning Jobs lifecycle API
- [x] Standard `/v1/images/generations` provider bridge
- [x] General idempotency framework for mutation APIs
- [x] Signed webhook delivery with retries and dead-letter handling

## Remaining priority work
- [x] Elastic FSDP checkpoint resharding
- [x] DeepSpeed/ZeRO trainer integration
- [x] Context/sequence/pipeline/expert-parallel trainer integration (PP uses stage partitioning + multi-microbatch 1F1B)
- [x] MoE tensor parallelism
- [ ] Large-MoE multi-node qualification
- [x] Production runtime qualification evidence runner (actual target hardware evidence still required)
- [x] Experiment tracking wired into every trainer
- [x] Full seq2seq dataset -> trainer -> checkpoint -> export -> serving lifecycle
- [x] TensorRT-LLM / FlashInfer / FlashAttention execution integration (hardware qualification remains)
- [ ] CUDA INT4/FP8 and native GPTQ/AWQ
- [x] Quantized llama.cpp-ready GGUF pipeline for the strict LLaMA-compatible MiniGPT subset (target-runtime qualification remains)
- [x] Async token-level vLLM streaming
- [x] Scalable ANN semantic-cache index + global distributed cache quotas
- [ ] Production multimodal gaps listed in `docs/CURRENT_MISSING_FEATURES_AUDIT.md`
- [x] Billing/entitlements, moderation, enterprise compliance, Kubernetes/autoscaling source implementation
- [x] Vector-store ingestion/search worker
- [x] Batch execution worker
- [x] Fine-tuning job orchestration worker
- [ ] Additional MCP transports and verifier sandbox isolation

- [x] Fine-tuning Jobs execution worker
- [x] SCIM/policy, signed audit, artifact admission, secret manager, KMS/HSM, rollout, backup/restore and multi-region source primitives
