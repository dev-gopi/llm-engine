# Phased Upgrade Roadmap (`docs/UPGRADE_ROADMAP.md`)

This roadmap breaks down the planned upgrades into 10 structured, priority-ranked development phases tailored to our architecture and 4 GB hardware constraints.

---

## Roadmap Overview

```text
Phase 0: Cleanup & Agent Documentation (CURRENT)
   ↓
Phase 1: Pretraining Data Scale, Deduplication & Quality Filtering
   ↓
Phase 2: Instruction Tuning (SFT) & Chat Formatting
   ↓
Phase 3: Reasoning & Preference Alignment (DPO)
   ↓
Phase 4: Efficient Inference & KV Cache Optimization
   ↓
Phase 5: Tool Calling, MCP & Autonomous Agent Workflows
   ↓
Phase 6: Long Context (2K -> 4K -> 8K) & RoPE Scaling
   ↓
Phase 7: Speculative Decoding & Advanced Generation
   ↓
Phase 8: Vision & Multimodal Integration
   ↓
Phase 9: Production Hardening & Scaled Architectures (MoE)
```

---

## Phase 0: Cleanup, Packaging & Agent Documentation System (CURRENT)

| Task ID | Description | Status | Dependencies | Files Affected | Expected Output | Validation Method | Risk | Priority |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **OPS-001** | Packaging & metadata cleanup (`egg-info`, stale logs, atomic scratch) | `IN_PROGRESS` | None | `src/llm_engine.egg-info`, `reports/`, `.gitignore` | Clean working tree; no tracked generated metadata | `git status`, test suite execution | Very Low | P0 |
| **OPS-002** | Create agent entry point & progressive context documentation system | `IN_PROGRESS` | OPS-001 | `AGENTS.md`, `docs/*.md`, `docs/context/` | Complete persistent engineering memory for AI agents | Verification of link validity and context completeness | Low | P0 |
| **OPS-003** | Create machine-friendly index & ADR baseline | `IN_PROGRESS` | OPS-002 | `docs/PROJECT_INDEX.md`, `docs/decisions/` | Fast component lookup and architecture decisions recorded | Manual audit against codebase | Low | P0 |

---

## Phase 1: Pretraining Data Scale, Deduplication & Quality Filtering

| Task ID | Description | Status | Dependencies | Files Affected | Expected Output | Validation Method | Risk | Priority |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DATA-001**| MinHash LSH deduplication pipeline | `PLANNED` | Phase 0 | `src/datasets/filters.py`, `scripts/audit_datasets.py` | Elimination of duplicate and near-duplicate training documents | `tests/test_dataset_governance.py` | Low | P1 |
| **DATA-002**| Multi-domain dataset expansion (FineWeb-Edu, Code, Math) | `PLANNED` | DATA-001 | `src/datasets/loader.py`, `configs/pretraining.gpu.yaml` | High-quality balanced dataset mix ready for streaming training | Shard token counts and quality audits | Medium | P1 |
| **DATA-003**| Dynamic curriculum and domain weighting | `PLANNED` | DATA-002 | `src/datasets/sampler.py`, `src/training/data.py` | Adaptive domain sampling schedules during pretraining | Loss curves across domains | Medium | P2 |

---

## Phase 2: Instruction Tuning (SFT) & Chat Formatting

| Task ID | Description | Status | Dependencies | Files Affected | Expected Output | Validation Method | Risk | Priority |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CHAT-001**| Special chat token stabilization & Jinja2 template engine | `PLANNED` | Phase 0 | `src/model/vocabulary.py`, `src/inference/chat_session.py` | Clean `<|system|>`, `<|user|>`, `<|assistant|>` chat formatting | `tests/test_chat_session.py` | Low | P1 |
| **CHAT-002**| Multi-turn instruction SFT pipeline with loss masking | `PLANNED` | CHAT-001 | `src/training/trainer.py`, `configs/finetuning.gpu.yaml` | Fine-tuned chat checkpoint answering multi-turn queries | Validation perplexity & retention tests | Medium | P1 |
| **CHAT-003**| Structured output & JSON schema constrained generation | `PLANNED` | CHAT-002 | `src/inference/generator.py`, `src/inference/sampler.py` | Output tokens strictly conforming to target JSON schemas | Schema validation unit tests | Medium | P2 |

---

## Phase 3: Reasoning & Preference Alignment (DPO)

| Task ID | Description | Status | Dependencies | Files Affected | Expected Output | Validation Method | Risk | Priority |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RSN-001** | Chain-of-thought (CoT) reasoning dataset integration | `PLANNED` | Phase 2 | `data/processed/`, `scripts/prepare_sft_stage.py` | Training corpus with explicit `<thinking>...</thinking>` traces | Inspection of reasoning data quality | Low | P1 |
| **RSN-002** | Direct Preference Optimization (DPO) training pipeline | `PLANNED` | RSN-001 | `src/post_training/dpo.py`, `scripts/train_dpo.py` | Aligned model preferring helpful/safe answers | `tests/test_dpo_workflow.py` | Medium | P1 |
| **RSN-003** | Mathematical & logic verification benchmarks | `PLANNED` | RSN-002 | `src/evaluation/benchmarks.py`, `scripts/evaluate_benchmarks.py` | Automated GSM8K and logic benchmark evaluation | Benchmark accuracy tracking | Low | P2 |

---

## Phase 4: Efficient Inference & KV Cache Optimization

| Task ID | Description | Status | Dependencies | Files Affected | Expected Output | Validation Method | Risk | Priority |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **INF-001** | Production Paged KV Cache with prefix caching | `PLANNED` | Phase 0 | `src/inference/paged_kv_cache.py`, `src/inference/generator.py` | Zero-allocation KV block reuse across shared prompts | KV cache memory profiling | Medium | P1 |
| **INF-002** | Continuous batching serving engine | `PLANNED` | INF-001 | `src/serving/batching.py`, `src/serving/runtime.py` | Iteration-level scheduling maximizing GPU utilization | `tests/test_serving.py`, load tests | Medium | P1 |
| **INF-003** | INT8 / INT4 weight & KV cache quantization | `PLANNED` | INF-001 | `src/inference/quantization.py`, `configs/inference.yaml` | 50% memory reduction with <1% perplexity degradation | `tests/test_inference_precision.py` | Medium | P2 |
| **INF-004** | Full OpenAI-compatible REST API endpoints | `PLANNED` | INF-002 | `src/serving/api.py`, `src/serving/schemas.py` | Standard `/v1/chat/completions` and `/v1/models` endpoints | Integration test with `httpx` | Low | P1 |

---

## Phase 5: Tool Calling, MCP & Autonomous Agent Workflows

| Task ID | Description | Status | Dependencies | Files Affected | Expected Output | Validation Method | Risk | Priority |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **AGT-001** | Tool schema parser & function call generation | `PLANNED` | Phase 2, Phase 4 | `src/inference/generator.py`, `src/inference/local_tools.py` | Model autonomously emits structured `<tool_call>` JSON | Tool syntax & argument tests | Medium | P1 |
| **AGT-002** | Multi-step agent execution loop with error recovery | `PLANNED` | AGT-001 | `src/serving/workspace.py`, `src/mcp/orchestration.py` | Full multi-turn Tool $\rightarrow$ Observation $\rightarrow$ Answer loop | `tests/test_workspace_agent.py` | Medium | P1 |
| **AGT-003** | Local vector RAG & document retrieval | `PLANNED` | AGT-002 | `src/inference/rag.py`, `scripts/build_rag_index.py` | Semantic search and document citation generation | `tests/test_rag.py` | Low | P2 |

---

## Phase 6: Long Context & Hybrid Attention

| Task ID | Description | Status | Dependencies | Files Affected | Expected Output | Validation Method | Risk | Priority |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CTX-001** | RoPE scaling (NTK-aware / YaRN) for 2K/4K/8K contexts | `PLANNED` | Phase 0 | `src/model/positional.py`, `configs/model.gpu.yaml` | Extended context inference without catastrophic perplexity explosion | Needle-in-a-haystack retrieval tests | Medium | P2 |
| **CTX-002** | Linear attention / recurrent attention sublayer | `PLANNED` | CTX-001 | `src/model/attention.py`, `src/model/transformer_block.py` | Sub-quadratic attention computation for long contexts | `tests/test_attention.py` | High | P3 |
| **CTX-003** | Hybrid architecture (75% linear + 25% full attention) | `PLANNED` | CTX-002 | `src/model/gpt.py`, `configs/model.gpu.yaml` | Hybrid attention backbone validated on 4 GB VRAM | Long-context training benchmark | High | P3 |

---

## Phase 7: Speculative Decoding & Advanced Generation

| Task ID | Description | Status | Dependencies | Files Affected | Expected Output | Validation Method | Risk | Priority |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SPC-001** | Multi-Token Prediction (MTP) auxiliary heads | `PLANNED` | Phase 0 | `src/model/gpt.py`, `src/training/trainer.py` | Simultaneous prediction of $k$ future tokens | Training loss convergence | Medium | P3 |
| **SPC-002** | Speculative decoding candidate acceptance engine | `PLANNED` | SPC-001, Phase 4 | `src/inference/generator.py` | 1.8x – 2.5x throughput speedup during generation | Tokens/sec benchmark comparisons | Medium | P3 |

---

## Phase 8: Vision & Multimodal Integration

| Task ID | Description | Status | Dependencies | Files Affected | Expected Output | Validation Method | Risk | Priority |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **VIS-001** | Vision encoder & spatial patch projection | `PLANNED` | Phase 0 | `src/vision/encoder.py`, `src/multimodal/projector.py` | Image embeddings projected into LLM token space | `tests/test_vision_models.py` | Medium | P3 |
| **VIS-002** | Multimodal instruction tuning pipeline | `PLANNED` | VIS-001, Phase 2 | `src/multimodal/model.py`, `scripts/train_vision.py` | Vision-language chat model handling image + text queries | Multimodal VQA evaluation | High | P3 |

---

## Phase 9: Production Hardening & Scaled Architectures (MoE)

| Task ID | Description | Status | Dependencies | Files Affected | Expected Output | Validation Method | Risk | Priority |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MOE-001** | Top-K sparse Mixture of Experts (MoE) FFN | `PLANNED` | Phase 0 | `src/model/feed_forward.py`, `src/model/config.py` | Sparse routing with auxiliary load-balancing loss | `tests/test_feed_forward.py` | High | P4 |
| **MOE-002** | Model registry & GGUF / ONNX export automation | `PLANNED` | Phase 4 | `scripts/export.py`, `docs/DEPLOYMENT.md` | Single-command deployment artifacts with SHA256 manifests | Export validation tests | Low | P2 |
| **SCALE-001**| Multi-GPU FSDP ZeRO-3 distributed pretraining validation | `PLANNED` | Phase 0 | `src/training/distributed.py`, `src/training/multinode.py` | Validated multi-GPU/multi-node pretraining pipeline | `tests/test_multinode.py`, `tests/test_scaling_features.py` | Medium | P1 |
| **SCALE-002**| Progressive model growth (80M → 1B) depth expansion | `PLANNED` | Phase 0 | `src/training/model_growth.py`, `scripts/grow_checkpoint.py` | Zero-divergence expanded checkpoint ready for continual pretraining | `tests/test_model_growth.py` | Medium | P2 |
| **SCALE-003**| Dynamic LoRA adapter serving & weight merging pipeline | `PLANNED` | Phase 4 | `src/training/peft.py`, `src/inference/generator.py` | Low-rank adapter swapping and standalone merged export | `tests/test_peft.py` | Low | P2 |

