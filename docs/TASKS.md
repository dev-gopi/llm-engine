# Persistent Agent Task System (`docs/TASKS.md`)

This task registry maintains stable task identifiers across sessions. When starting work on any task, update its status from `TODO` to `IN_PROGRESS`, and upon completion and test verification, mark it `COMPLETED` and update [`docs/CHANGELOG.md`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/CHANGELOG.md).

---

## Active & Upcoming Tasks

### OPS-001: Packaging & Metadata Cleanup
- **ID**: `OPS-001`
- **Title**: Packaging & Metadata Cleanup
- **Status**: `COMPLETED`
- **Priority**: `P0`
- **Description**: Audit `src/llm_engine.egg-info`, stale training report fragments, and clean temporary files from the repository.
- **Why**: Prevent build metadata drift, ensure a clean working tree, and eliminate untracked scratch artifacts.
- **Dependencies**: None
- **Relevant files**:
  - `src/llm_engine.egg-info`
  - `reports/.finetuning.json.a_l4qm6d`
  - `.gitignore`
- **Implementation notes**: `src/llm_engine.egg-info` was created by `pip install -e .` due to `pyproject.toml` `where = ["src"]`. It is gitignored and safe to remove.
- **Validation**: `git status` shows clean working tree; pytest passes cleanly.
- **Acceptance criteria**: All temporary scratch files removed, no generated files tracked in Git.

---

### OPS-002: AI Agent Documentation & Knowledge System
- **ID**: `OPS-002`
- **Title**: AI Agent Documentation & Engineering Knowledge System
- **Status**: `COMPLETED`
- **Priority**: `P0`
- **Description**: Build progressive context documentation system with root `AGENTS.md`, machine index, subsystem guides, compact context packs, and ADRs.
- **Why**: Enable any AI coding agent to onboard immediately without scanning the entire repository, conserving context tokens.
- **Dependencies**: `OPS-001`
- **Relevant files**:
  - `AGENTS.md`
  - `docs/*.md`
  - `docs/context/*.context.md`
  - `docs/decisions/*.md`
- **Implementation notes**: Keep context packs under 200 lines and link to authoritative source files.
- **Validation**: All documentation files exist, cross-links resolve, and guidelines match codebase reality.
- **Acceptance criteria**: A complete documentation suite acts as persistent project memory.

---

### DATA-001: MinHash Deduplication Pipeline
- **ID**: `DATA-001`
- **Title**: Scalable MinHash LSH Corpus Deduplication
- **Status**: `COMPLETED`
- **Priority**: `P1`
- **Description**: Implement MinHash Locality-Sensitive Hashing (LSH) deduplication in the data preprocessing pipeline.
- **Why**: Pretraining on repetitive web data degrades model quality and causes memorization.
- **Dependencies**: `OPS-002`
- **Relevant files**:
  - `src/datasets/filters.py`
  - `scripts/clean_jsonl_corpus.py`
  - `tests/test_clean_jsonl_corpus.py`
- **Implementation notes**: Compute n-gram shingles (n=5) and 128 hash permutations per document. Group into bands for collision detection.
- **Validation**: Deduplication test verifies removal of 90%+ similar documents while retaining distinct texts.
- **Acceptance criteria**: Dataset preprocessing script outputs clean corpus with duplicate metrics reported.

---

### CHAT-001: Chat Template Standardization & Loss Masking
- **ID**: `CHAT-001`
- **Title**: Standardized Chat Templating with Prompt Loss Masking
- **Status**: `COMPLETED`
- **Priority**: `P1`
- **Description**: Standardize multi-turn chat templates (`<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>`) and apply cross-entropy loss only on assistant tokens.
- **Why**: Base models learn to generate prompts rather than answers if prompt tokens are unmasked during fine-tuning.
- **Dependencies**: `OPS-002`
- **Relevant files**:
  - `src/inference/chat_session.py`
  - `src/datasets/collator.py`
  - `src/training/trainer.py`
  - `tests/test_chat_session.py`
- **Implementation notes**: Target tokens corresponding to system and user turns should be assigned `ignore_index` (-100).
- **Validation**: Verify backward pass gradients are zero for prompt token embeddings during SFT.
- **Acceptance criteria**: `Trainer` only penalizes next-token errors for assistant responses.

---

### RSN-001: Chain-of-Thought Reasoning Pipeline
- **ID**: `RSN-001`
- **Title**: Structured Chain-of-Thought Reasoning with Thinking Tags
- **Status**: `COMPLETED`
- **Priority**: `P1`
- **Description**: Add `<thinking>...</thinking>` token traces to fine-tuning data and evaluate reasoning performance on GSM8K.
- **Why**: Enables step-by-step problem solving for arithmetic and algorithmic questions.
- **Dependencies**: `CHAT-001`
- **Relevant files**:
  - `src/model/vocabulary.py`
  - `scripts/prepare_sft_stage.py`
  - `src/evaluation/benchmarks.py`
- **Implementation notes**: The thinking tokens must be trained explicitly with CoT data; token presence alone does not induce reasoning.
- **Validation**: Benchmark evaluation on GSM8K subset matches or exceeds baseline accuracy.
- **Acceptance criteria**: Model produces verified structured reasoning traces before providing final answers.

---

### INF-001: Production Paged KV Cache with Prefix Caching
- **ID**: `INF-001`
- **Title**: Block-Paged KV Cache with System Prompt Prefix Sharing
- **Status**: `COMPLETED`
- **Priority**: `P1`
- **Description**: Implement virtual block-based KV cache allocation with hash-indexed prefix caching across concurrent sessions.
- **Why**: Eliminates memory fragmentation and prevents recomputing KV representations for repeated system prompts or tool schemas.
- **Dependencies**: `OPS-002`
- **Relevant files**:
  - `src/inference/paged_kv_cache.py`
  - `src/inference/generator.py`
  - `tests/test_generation.py`
- **Implementation notes**: Block size should be 16 or 32 tokens. Shared prefix blocks are marked read-only with reference counting.
- **Validation**: Prefix caching benchmark shows 0 ms prefill latency for repeated identical prompt prefixes.
- **Acceptance criteria**: Prefill time for cached system prompts drops by >90% without output deviation.

---

### AGT-001: Constrained JSON Tool Calling & Schema Validation
- **ID**: `AGT-001`
- **Title**: Constrained JSON Tool Calling & Schema Validation
- **Status**: `COMPLETED`
- **Priority**: `P1`
- **Description**: Add grammar-constrained decoding and strict JSON schema validation for tool function calls.
- **Why**: LLMs often generate slightly malformed JSON, causing runtime crashes in agent loops.
- **Dependencies**: `CHAT-001`, `INF-001`
- **Relevant files**:
  - `src/inference/generator.py`
  - `src/inference/local_tools.py`
  - `src/mcp/client.py`
  - `tests/test_local_tools.py`
- **Implementation notes**: Enforce JSON state machine or logit masking during tool invocation generation.
- **Validation**: 100 tool calls tested with 0 syntax parsing errors.
- **Acceptance criteria**: Model reliably selects tools, populates validated JSON arguments, and consumes results.

---

### CTX-001: Long Context RoPE Scaling (NTK & YaRN)
- **ID**: `CTX-001`
- **Title**: Context Extension via NTK-Aware & YaRN RoPE Scaling
- **Status**: `COMPLETED`
- **Priority**: `P2`
- **Description**: Implement NTK-aware and YaRN position interpolation in `RotaryEmbedding` to expand context to 2K, 4K, and 8K.
- **Why**: Current 1024 context window restricts multi-turn chat and document RAG.
- **Dependencies**: `OPS-002`
- **Relevant files**:
  - `src/model/positional.py`
  - `configs/model.gpu.yaml`
  - `tests/test_positional.py`
- **Implementation notes**: Compute wavelength-dependent scaling factors to preserve high-frequency features while expanding lower frequencies.
- **Validation**: Synthetic passkey retrieval ("Needle in a Haystack") at 2048 and 4096 tokens.
- **Acceptance criteria**: >95% passkey retrieval accuracy across the full extended context window.

---

### SCALE-001: FSDP ZeRO-3 Production Validation & Benchmark
- **ID**: `SCALE-001`
- **Title**: Multi-GPU FSDP ZeRO-3 Distributed Pretraining Validation
- **Status**: `IN_PROGRESS`
- **Priority**: `P1`
- **Description**: Validate FSDP full parameter and optimizer sharding on multi-GPU nodes with `configs/text/pretraining.future.fsdp.yaml`.
- **Why**: Scaling beyond 1B parameters requires parameter and gradient sharding across multiple accelerators.
- **Dependencies**: `OPS-002`
- **Relevant files**:
  - `src/training/distributed.py`
  - `src/training/multinode.py`
  - `src/training/distributed_checkpoint.py`
  - `tests/test_multinode.py`
  - `tests/test_scaling_features.py`
- **Implementation notes**: Verify rank-parallel sharded checkpoint saves and resume integrity across dynamic worker restarts.
- **Validation**: 2-node 4-GPU distributed test run with zero rank desynchronization.
- **Acceptance criteria**: Gradient and loss parity between single-GPU and 4-GPU FSDP runs within numerical epsilon.

---

### SCALE-002: Model Growth (80M → 1B) Checkpoint Expansion
- **ID**: `SCALE-002`
- **Title**: Progressive Model Depth & Width Expansion Pipeline
- **Status**: `IN_PROGRESS`
- **Priority**: `P2`
- **Description**: Expand trained 80M base checkpoint to 1B architecture using `src/training/model_growth.py` with identity residual initialization.
- **Why**: Warms up larger architectures 3x faster than training from random initializations.
- **Dependencies**: `OPS-002`
- **Relevant files**:
  - `src/training/model_growth.py`
  - `scripts/grow_checkpoint.py`
  - `tests/test_model_growth.py`
- **Implementation notes**: Initialize new layer attention output projections and MLP down-projections with exact zeros so initial forward output is identical.
- **Validation**: `tests/test_model_growth.py` verifies output divergence is zero at initialization step.
- **Acceptance criteria**: Successfully grow checkpoint from 16 to 32 layers with monotonic loss descent in subsequent pretraining.

---

### SCALE-003: PEFT/LoRA Adapter Serving & Dynamic Merging
- **ID**: `SCALE-003`
- **Title**: Dynamic LoRA Adapter Swapping and Base Weight Merging
- **Status**: `COMPLETED`
- **Priority**: `P2`
- **Description**: Add runtime LoRA adapter swapping in `AutoregressiveGenerator` and offline `merge_and_unload()` export script.
- **Why**: Allows serving multiple task-specific domain adapters from a single shared base model in 4 GB VRAM.
- **Dependencies**: `INF-001`
- **Relevant files**:
  - `src/training/peft.py`
  - `src/inference/generator.py`
  - `scripts/export.py`
  - `tests/test_peft.py`
- **Implementation notes**: Weight merging formula: $W_{merged} = W_{base} + \frac{\alpha}{r} (B \times A)$.
- **Validation**: Unit test verifies output of merged model matches model with adapter active.
- **Acceptance criteria**: Standalone merged checkpoint loads without PEFT dependency.

---

### VIS-001: Vision-Language Model Projector Pretraining
- **ID**: `VIS-001`
- **Title**: VisionProjector Alignment Training with Frozen Backbones
- **Status**: `COMPLETED`
- **Priority**: `P2`
- **Description**: Train `VisionProjector` connecting `VisionEncoder` and `MiniGPT` on image-caption pairs while freezing both backbones.
- **Why**: Establishes cross-modal alignment without catastrophic forgetting of text generation capabilities.
- **Dependencies**: `OPS-002`
- **Relevant files**:
  - `src/multimodal/model.py`
  - `src/multimodal/projector.py`
  - `configs/vision/multimodal.yaml`
  - `tests/test_vision_models.py`
- **Implementation notes**: Frozen backbones must be set to `eval()` mode to ensure deterministic representations without dropout noise.
- **Validation**: Training loss convergence on multimodal image-caption split.
- **Acceptance criteria**: `VisionLanguageModel` generates coherent descriptive text from input images.

---

## Audited Backlog

The following tasks were added during the documentation audit on 2026-09-20. They replace ambiguous roadmap-only items with stable IDs, authoritative files, and testable scope. Hardware- or dataset-dependent work remains explicitly separate from local implementation work.

### TKN-001: Append-Only Agent Protocol Tokens
- **ID**: `TKN-001`
- **Status**: `COMPLETED`
- **Priority**: `P1`
- **Description**: Add a verified append-only extension for `<|system|>`, `<|user|>`, `<|assistant|>`, `<|tool|>`, `<|thinking|>`, and `<|end|>`.
- **Why**: Agent/chat delimiters must be atomic special tokens without reindexing existing checkpoints.
- **Dependencies**: `CHAT-001`, `RSN-001`, `AGT-001`
- **Relevant files**: `src/tokenizer/encoder.py`, `src/model/vocabulary.py`, `tests/test_tokenizer.py`, `tests/test_vocabulary_compatibility.py`
- **Validation**: `.venv/bin/pytest tests/test_tokenizer.py tests/test_vocabulary_compatibility.py -q`
- **Acceptance criteria**: All six tokens are atomic, survive save/load, and preserve every base vocabulary ID.

### EVAL-001: Long-Context Needle-in-a-Haystack Evaluation
- **ID**: `EVAL-001`
- **Status**: `COMPLETED`
- **Priority**: `P1`
- **Description**: Add a deterministic synthetic passkey-retrieval benchmark covering 2K and 4K contexts, with explicit needle placement and exact-answer scoring.
- **Why**: RoPE scaling is implemented, but no repeatable harness establishes whether an extended-context checkpoint can retrieve information across its claimed window.
- **Dependencies**: `CTX-001`
- **Relevant files**: `src/evaluation/benchmarks.py`, `tests/test_evaluation_regressions.py`
- **Validation**: `.venv/bin/pytest tests/test_evaluation_regressions.py -q`
- **Acceptance criteria**: The harness produces deterministic, correctly positioned 2K/4K prompts and rejects incorrect passkeys.

### RSN-003: Reasoning and Code Evaluation Suite
- **ID**: `RSN-003`
- **Status**: `COMPLETED`
- **Priority**: `P2`
- **Description**: Add versioned math, logic, and code benchmark manifests plus a comparison report format.
- **Dependencies**: `RSN-001`
- **Relevant files**: `src/evaluation/benchmarks.py`, `scripts/evaluate_benchmarks.py`, `tests/test_evaluation_regressions.py`
- **Validation**: `.venv/bin/pytest tests/test_evaluation_regressions.py -q`
- **Acceptance criteria**: Versioned reasoning/code probes preserve all scoring controls and produce comparable regression reports.

### INF-002: Continuous-Batching Execution Engine
- **ID**: `INF-002`
- **Status**: `COMPLETED`
- **Priority**: `P1`
- **Description**: Evolve request queuing into iteration-level continuous batching with correct request admission, completion, and cache ownership.
- **Dependencies**: `INF-001`
- **Relevant files**: `src/serving/batching.py`, `src/inference/paged_kv_cache.py`, `tests/test_serving.py`

### INF-005: Paged-Attention Execution Integration
- **ID**: `INF-005`
- **Status**: `COMPLETED`
- **Priority**: `P2`
- **Description**: Replace materialized per-request KV views in the serving decode path with page-table-aware attention execution.
- **Why**: The allocator and prefix cache account for pages, but materializing a contiguous cache per request does not provide full paged-attention memory behavior.
- **Dependencies**: `INF-001`, `INF-002`
- **Relevant files**: `src/inference/paged_kv_cache.py`, `src/inference/generator.py`, `src/model/attention.py`, `src/model/transformer_block.py`, `src/model/gpt.py`, `tests/test_generation.py`, `tests/test_attention.py`
- **Validation**: `.venv/bin/pytest tests/test_generation.py tests/test_serving.py -q`
- **Acceptance criteria**: Batched decode consumes page tables without reconstructing a contiguous KV tensor for every active request. Verified against contiguous-cache attention output.

### AGT-002: Permissioned Multi-Step Tool Orchestration
- **ID**: `AGT-002`
- **Status**: `COMPLETED`
- **Priority**: `P1`
- **Description**: Implement an explicit Tool → Observation → Answer loop with per-tool permissions, bounded retries, and auditable failures.
- **Dependencies**: `AGT-001`
- **Relevant files**: `src/mcp/client.py`, `src/mcp/orchestration.py`, `src/inference/local_tools.py`, `src/serving/workspace.py`, `src/serving/backend.py`, `tests/test_mcp_client.py`, `tests/test_local_tools.py`, `tests/test_workspace_agent.py`
- **Validation**: `.venv/bin/pytest tests/test_mcp_client.py tests/test_local_tools.py tests/test_workspace_agent.py tests/test_rag.py -q`
- **Acceptance criteria**: Sequential calls are bounded, allowlisted, schema-validated, and tool failures remain structured model context.

### RAG-002: Retrieval Reranking and Citations
- **ID**: `RAG-002`
- **Status**: `COMPLETED`
- **Priority**: `P2`
- **Description**: Add deterministic reranking, context-budget construction, and source citations to retrieved answers.
- **Dependencies**: `AGT-002`
- **Relevant files**: `src/inference/rag.py`, `tests/test_rag.py`
- **Validation**: `.venv/bin/pytest tests/test_rag.py -q`
- **Acceptance criteria**: Retrieval results are deterministically reranked, total context is bounded, and every citation includes a source URL.

### DATA-002: Reproducible Multi-Domain Data Mixtures
- **ID**: `DATA-002`
- **Status**: `COMPLETED`
- **Priority**: `P2`
- **Description**: Define versioned, license-aware streaming mixtures for web, code, math, and reasoning data; retain per-source quality metrics.
- **Dependencies**: `DATA-001`
- **Relevant files**: `src/datasets/loader.py`, `configs/pretraining.gpu.yaml`, `tests/test_dataset_governance.py`
- **Validation**: `.venv/bin/pytest tests/test_dataset_governance.py -q` (9 passed)
- **Acceptance criteria**: Source declarations require unique name, domain, version, paths, positive weight, license, and numeric quality metrics; records stream in deterministic weighted order with source/version provenance and no corpus-wide materialization.

### DATA-003: Curated Capability Corpus Acquisition and Audit
- **ID**: `DATA-003`
- **Status**: `TODO`
- **Priority**: `P0`
- **Description**: Curate reviewed, versioned web, documentation, code, mathematics, science, multilingual, conversation, instruction, reasoning, and tool-use corpora; publish provenance, licensing, quality, and contamination audit artifacts before activation.
- **Dependencies**: `DATA-002`
- **Relevant files**: `configs/pretraining.gpu.yaml`, `src/datasets/governance.py`, `scripts/audit_datasets.py`, `tests/test_dataset_governance.py`

### CHAT-002: Instruction Quality and Multi-Turn Evaluation
- **ID**: `CHAT-002`
- **Status**: `TODO`
- **Priority**: `P0`
- **Description**: Define a reviewed instruction/multi-turn corpus contract covering clarification, refusals, format following, conversational consistency, and tool-message turns; add deterministic instruction-following evaluation.
- **Dependencies**: `CHAT-001`, `DATA-003`
- **Relevant files**: `src/inference/chat_session.py`, `src/evaluation/benchmarks.py`, `tests/test_chat_session.py`, `tests/test_evaluation_regressions.py`

### RSN-002: Reasoning SFT Data and Training Profile
- **ID**: `RSN-002`
- **Status**: `TODO`
- **Priority**: `P0`
- **Description**: Add governed math, code, logic, planning, verification, and self-correction reasoning SFT data plus a reproducible training profile that preserves explicit reasoning-trace boundaries.
- **Dependencies**: `RSN-001`, `DATA-003`
- **Relevant files**: `src/inference/chat_session.py`, `src/training/trainer.py`, `configs/finetuning.gpu.yaml`, `tests/test_chat_session.py`, `tests/test_training_system.py`

### CTX-002: Long-Context Training and Retrieval Validation
- **ID**: `CTX-002`
- **Status**: `TODO`
- **Priority**: `P0`
- **Description**: Produce 2K/4K/8K context training profiles, long-document data packs, memory measurements, and checkpoint-backed retrieval evaluations before claiming long-context capability.
- **Dependencies**: `CTX-001`, `EVAL-001`, `DATA-003`
- **Relevant files**: `configs/pretraining.gpu.yaml`, `src/model/positional.py`, `scripts/evaluate_benchmarks.py`, `tests/test_positional.py`, `tests/test_evaluation_regressions.py`

### AGT-003: Tool-Use Training and Reliability Evaluation
- **ID**: `AGT-003`
- **Status**: `TODO`
- **Priority**: `P1`
- **Description**: Build governed tool-use SFT/evaluation cases for selection, schema-conformant arguments, result handling, error recovery, sequential/parallel calls, and permission boundaries.
- **Dependencies**: `AGT-001`, `AGT-002`, `CHAT-002`
- **Relevant files**: `src/inference/generator.py`, `src/inference/local_tools.py`, `src/evaluation/benchmarks.py`, `tests/test_generation.py`, `tests/test_local_tools.py`

### EVAL-002: Comprehensive Capability Regression Matrix
- **ID**: `EVAL-002`
- **Status**: `TODO`
- **Priority**: `P0`
- **Description**: Version a release-gating evaluation matrix for knowledge, math, code, reasoning, instruction following, structured JSON, tools, RAG, long context, safety, hallucination, and refusal behavior.
- **Dependencies**: `RSN-003`, `SAFE-001`, `EVAL-001`, `CHAT-002`, `AGT-003`, `EMB-001`
- **Relevant files**: `src/evaluation/benchmarks.py`, `scripts/evaluate_benchmarks.py`, `configs/evaluation.domains.yaml`, `tests/test_evaluation_regressions.py`

### VIS-002: Multimodal Instruction-Tuning Pipeline
- **ID**: `VIS-002`
- **Status**: `TODO`
- **Priority**: `P3`
- **Description**: Add image-text SFT data handling and evaluation for the frozen-backbone projector profile.
- **Dependencies**: `VIS-001`, `CHAT-001`
- **Relevant files**: `src/multimodal/model.py`, `src/vision/encoder.py`, `tests/test_vision_models.py`

### SPC-001: Speculative Decoding Baseline
- **ID**: `SPC-001`
- **Status**: `TODO`
- **Priority**: `P3`
- **Description**: Implement draft/target candidate verification and a reproducible throughput benchmark before adding multi-token prediction training.
- **Dependencies**: `INF-002`
- **Relevant files**: `src/inference/generator.py`, `tests/test_generation.py`

### MTP-001: Multi-Token Prediction Training Objective
- **ID**: `MTP-001`
- **Status**: `TODO`
- **Priority**: `P3`
- **Description**: Add an opt-in multi-token prediction auxiliary objective and validate its compatibility with ordinary causal-language-model checkpoints before speculative-decoding integration.
- **Dependencies**: `SPC-001`
- **Relevant files**: `src/model/gpt.py`, `src/model/loss.py`, `src/training/trainer.py`, `tests/test_gpt.py`, `tests/test_loss.py`

### OUT-001: Token-Level Structured Output Constraints
- **ID**: `OUT-001`
- **Status**: `TODO`
- **Priority**: `P2`
- **Description**: Add token-time JSON/grammar constraints for supported schemas; preserve post-generation validation as a defense in depth layer.
- **Dependencies**: `AGT-001`
- **Relevant files**: `src/inference/generator.py`, `src/inference/sampler.py`, `tests/test_generation.py`

### QNT-001: Portable Low-Precision Export and KV-Cache Quantization
- **ID**: `QNT-001`
- **Status**: `COMPLETED`
- **Priority**: `P2`
- **Description**: Add reproducible FP16/BF16 and INT4 export choices plus measured KV-cache precision trade-offs.
- **Dependencies**: `INF-002`
- **Relevant files**: `src/inference/quantization.py`, `src/inference/paged_kv_cache.py`, `scripts/export.py`, `tests/test_inference_precision.py`
- **Validation**: `.venv/bin/pytest tests/test_inference_precision.py tests/test_export.py tests/test_scaling_features.py tests/test_attention.py -q` (51 passed)
- **Acceptance criteria**: Export supports FP16, BF16, and self-describing packed INT4 safetensors; per-token scaled INT8 paged-KV storage reports its full allocated footprint and dequantizes within tested error bounds.

### MEM-001: Privacy-Bounded Conversation Memory
- **ID**: `MEM-001`
- **Status**: `COMPLETED`
- **Priority**: `P2`
- **Description**: Add opt-in session memory with expiry, retrieval, deletion, and privacy boundaries suitable for agent use.
- **Dependencies**: `AGT-002`
- **Relevant files**: `src/inference/chat_session.py`, `src/serving/api.py`, `tests/test_chat_session.py`, `tests/test_serving.py`
- **Validation**: `.venv/bin/pytest tests/test_chat_session.py tests/test_serving.py -q` (46 passed)
- **Acceptance criteria**: Session-scoped lexical retrieval is bounded and never spans sessions; persisted history remains TTL-expiring; authenticated memory retrieval/deletion is disabled by default and requires explicit enablement.

### SEC-001: Serving Authentication, Tool Isolation, and Audit Events
- **ID**: `SEC-001`
- **Status**: `COMPLETED`
- **Priority**: `P1`
- **Description**: Close production-hardening gaps with explicit authorization policy, sandboxed tool execution, secret isolation, and immutable audit events.
- **Dependencies**: `AGT-002`
- **Relevant files**: `src/serving/api.py`, `src/mcp/client.py`, `src/inference/local_tools.py`, `tests/test_serving.py`, `tests/test_mcp_client.py`
- **Validation**: `.venv/bin/pytest tests/test_serving.py tests/test_mcp_client.py tests/test_local_tools.py -q`
- **Acceptance criteria**: Protected operations require authorization; sandbox/allowlist boundaries are retained; audit events exclude prompts, credentials, and tool payloads.

### REG-001: Versioned Model Registry and Deployment Manifests
- **ID**: `REG-001`
- **Status**: `COMPLETED`
- **Priority**: `P2`
- **Description**: Produce content-addressed model/export manifests with tokenizer, config, evaluation, and compatibility metadata.
- **Dependencies**: `QNT-001`
- **Relevant files**: `scripts/export.py`, `src/training/checkpoint.py`, `tests/test_export.py`
- **Validation**: `.venv/bin/pytest tests/test_export.py -q` (3 passed)
- **Acceptance criteria**: Every CLI export writes a relocatable `manifest.json` with a schema version, artifact/config/tokenizer SHA-256 hashes, tokenizer fingerprint, model configuration, export format, and weight precision.

### ATT-001: Hardware-Aware Efficient Attention Backend
- **ID**: `ATT-001`
- **Status**: `COMPLETED`
- **Priority**: `P2`
- **Description**: Select and validate a scaled-dot-product/FlashAttention backend by device and dtype, with a deterministic fallback for CPU and unsupported GPUs.
- **Why**: The architecture uses standard attention; long-context training and high-throughput decode need an explicit kernel-selection and validation contract.
- **Dependencies**: `CTX-001`, `INF-005`
- **Relevant files**: `src/model/attention.py`, `src/model/config.py`, `tests/test_attention.py`
- **Validation**: `.venv/bin/pytest tests/test_attention.py tests/test_model_config.py -q` (34 passed)
- **Acceptance criteria**: `attention_backend: auto` uses PyTorch SDPA on compatible CUDA FP16/BF16/FP32 tensors (allowing its FlashAttention dispatch) and the explicit eager implementation on CPU or unsupported inputs; `eager` always selects the deterministic fallback.

### SAFE-001: Safety, Jailbreak, and Prompt-Injection Evaluation
- **ID**: `SAFE-001`
- **Status**: `COMPLETED`
- **Priority**: `P1`
- **Description**: Add a versioned safety probe suite for refusal, jailbreak resistance, and untrusted tool/RAG prompt-injection handling.
- **Dependencies**: `AGT-002`, `EVAL-001`
- **Relevant files**: `src/inference/prompt_safety.py`, `src/evaluation/benchmarks.py`, `tests/test_prompt_safety.py`, `tests/test_evaluation_regressions.py`
- **Validation**: `.venv/bin/pytest tests/test_prompt_safety.py tests/test_evaluation_regressions.py -q`
- **Acceptance criteria**: A versioned deterministic manifest covers harmful, injection, and benign controls without retaining prompt contents in the summary.

### OBS-001: Serving Latency and Resource Observability
- **ID**: `OBS-001`
- **Status**: `COMPLETED`
- **Priority**: `P2`
- **Description**: Add bounded percentile latency metrics, queue delay, token throughput, error rate, and paged-KV utilization to the serving metrics contract.
- **Dependencies**: `INF-002`, `INF-005`
- **Relevant files**: `src/serving/runtime.py`, `src/serving/api.py`, `tests/test_serving.py`
- **Validation**: `.venv/bin/pytest tests/test_serving.py tests/test_serving_orchestration.py -q`
- **Acceptance criteria**: Metrics expose bounded latency percentiles, queue delay, token throughput, error rate, and paged-KV utilization when available.

### FTL-001: Checkpoint Integrity and Recovery Drills
- **ID**: `FTL-001`
- **Status**: `TODO`
- **Priority**: `P2`
- **Description**: Verify corrupt-checkpoint detection and deterministic restart/recovery of model, optimizer, scheduler, RNG, and sampler state.
- **Dependencies**: `SCALE-001`
- **Relevant files**: `src/training/checkpoint.py`, `src/training/distributed_checkpoint.py`, `tests/test_training_system.py`, `tests/test_multinode.py`

### CUR-001: Curriculum Scheduling and Gradient Diagnostics
- **ID**: `CUR-001`
- **Status**: `TODO`
- **Priority**: `P2`
- **Description**: Add configuration-driven domain curricula, per-domain loss reporting, activation/gradient diagnostics, and safe schedule resume semantics.
- **Dependencies**: `DATA-002`
- **Relevant files**: `src/datasets/sampler.py`, `src/training/trainer.py`, `configs/pretraining.gpu.yaml`, `tests/test_training_data.py`

### EMB-001: Dedicated Embedding and Retrieval Quality Evaluation
- **ID**: `EMB-001`
- **Status**: `TODO`
- **Priority**: `P2`
- **Description**: Define a dedicated embedding-model interface and retrieval recall/reranking evaluation rather than relying solely on lexical or generic model representations.
- **Dependencies**: `RAG-002`
- **Relevant files**: `src/inference/rag.py`, `src/evaluation/benchmarks.py`, `tests/test_rag.py`
