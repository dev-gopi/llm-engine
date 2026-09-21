# Persistent Agent Task System (`docs/TASKS.md`)

This task registry maintains stable task identifiers across sessions. When starting work on any task, update its status from `TODO` to `IN_PROGRESS`, and upon completion and test verification, mark it `COMPLETED` and update [`docs/CHANGELOG.md`]\(file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/CHANGELOG.md).

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

- **Description**: Add `\<thinking>...\</thinking>` token traces to fine-tuning data and evaluate reasoning performance on GSM8K.

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

- **Status**: `COMPLETED`

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

- **Status**: `COMPLETED`

- **Priority**: `P2`

- **Description**: Expand trained 80M base checkpoint to 1B architecture using `src/training/model_growth.py` with identity residual initialization.

- **Why**: Warms up larger architectures 3x faster than training from random initializations.

- **Dependencies**: `OPS-002`

- **Relevant files**:
  - `src/training/model_growth.py`

  - `scripts/grow_checkpoint.py`

  - `tests/test_model_growth.py`

- **Implementation notes**: Initialize new layer attention output projections and MLP down-projections with exact zeros so initial forward output is identical.

- **Validation**: `.venv/bin/pytest tests/test_scaling_features.py tests/test_model_growth.py tests/test_peft.py tests/test_tensor_parallel.py tests/test_multinode.py -q` (34 passed). `tests/test_model_growth.py` verifies zero output divergence for a 16-to-32-layer expansion at initialization.

- **Acceptance criteria**: Successfully grow a checkpoint from 16 to 32 layers with zero initialization-time output divergence. Monotonic continued-pretraining loss descent requires an external trained-checkpoint run and remains an operational validation requirement.

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

- **Implementation notes**: Weight merging formula: $W\_{merged} = W\_{base} + *\frac*{\alpha}{r} (B \times A)$.

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

- **Status**: `COMPLETED`

- **Priority**: `P0`

- **Description**: Curate reviewed, versioned web, documentation, code, mathematics, science, multilingual, conversation, instruction, reasoning, and tool-use corpora; publish provenance, licensing, quality, and contamination audit artifacts before activation.

- **Dependencies**: `DATA-002`

- **Relevant files**: `configs/pretraining.gpu.yaml`, `src/datasets/governance.py`, `scripts/audit_datasets.py`, `tests/test_dataset_governance.py`

### CHAT-002: Instruction Quality and Multi-Turn Evaluation

- **ID**: `CHAT-002`

- **Status**: `COMPLETED`

- **Priority**: `P0`

- **Description**: Define a reviewed instruction/multi-turn corpus contract covering clarification, refusals, format following, conversational consistency, and tool-message turns; add deterministic instruction-following evaluation.

- **Dependencies**: `CHAT-001`, `DATA-003`

- **Relevant files**: `src/inference/chat_session.py`, `src/evaluation/benchmarks.py`, `tests/test_chat_session.py`, `tests/test_evaluation_regressions.py`

- **Validation**: `.venv/bin/pytest tests/test_chat_session.py tests/test_evaluation_regressions.py -q` (18 passed).

- **Acceptance criteria**: Canonical SFT rendering accepts untrusted tool turns without supervising them; the versioned deterministic instruction suite covers clarification, safe refusal, exact-format following, conversational consistency, and tool-result handling. External corpus activation remains subject to dataset-manifest review.

### RSN-002: Reasoning SFT Data and Training Profile

- **ID**: `RSN-002`

- **Status**: `COMPLETED`

- **Priority**: `P0`

- **Description**: Add governed math, code, logic, planning, verification, and self-correction reasoning SFT data plus a reproducible training profile that preserves explicit reasoning-trace boundaries.

- **Dependencies**: `RSN-001`, `DATA-003`

- **Relevant files**: `src/inference/chat_session.py`, `src/training/trainer.py`, `configs/finetuning.gpu.yaml`, `tests/test_chat_session.py`, `tests/test_training_system.py`

- **Implementation notes**: Added six repository-authored reasoning SFT seed domains (math, code, logic, planning, verification, self-correction), a deterministic `configs/finetuning.reasoning.gpu.yaml` profile, strict `<thinking>...</thinking>` trace validation, and an `assistant_only` trainer policy requiring explicit loss masks for reasoning SFT.

- **Validation**: `python -m pytest tests/test_chat_session.py tests/test_training_system.py -q` (55 passed). The full suite was attempted but collection is blocked in this environment because `pyarrow` is unavailable and the environment cannot reach PyPI to install it.

- **Acceptance criteria**: Reasoning records have explicit trace/final-answer boundaries; prompt tokens are excluded from the reasoning SFT objective; the profile fixes seed, domain weights, stage ID, governance policy, and checkpoint paths; each reasoning domain has a manifest and train/validation split.

### CTX-002: Long-Context Training and Retrieval Validation

- **ID**: `CTX-002`

- **Status**: `COMPLETED`

- **Priority**: `P0`

- **Description**: Produce 2K/4K/8K context training profiles, long-document data packs, memory measurements, and checkpoint-backed retrieval evaluations before claiming long-context capability.

- **Dependencies**: `CTX-001`, `EVAL-001`, `DATA-003`

- **Relevant files**: `configs/pretraining.gpu.yaml`, `src/model/positional.py`, `scripts/evaluate_benchmarks.py`, `tests/test_positional.py`, `tests/test_evaluation_regressions.py`

- **Implementation notes**: Added paired 2K/4K/8K model and pretraining profiles, deterministic long-document retrieval fixtures with DATA-003-compatible manifests, checkpoint-backed long-context benchmark mode, and CUDA peak-memory measurement tooling. Wired `rope_scaling_type` and `rope_original_max_position` from model configuration into `MiniGPT`.

- **Validation**: `python -m pytest tests/test_positional.py tests/test_evaluation_regressions.py -q` (30 passed); DATA-003 audit of all three long-context fixture manifests passes. Full checkpoint-backed retrieval and GPU memory evidence remain pending until a matching 2K/4K/8K checkpoint is supplied and executed on the target GPU.

- **Acceptance criteria**: Profiles and governed fixtures are reproducible; benchmark and memory tools emit machine-readable evidence; long-context capability is not considered validated unless the requested checkpoint passes retrieval probes and memory measurements at the claimed context length.

### AGT-003: Tool-Use Training and Reliability Evaluation

- **ID**: `AGT-003`

- **Status**: `COMPLETED`

- **Priority**: `P1`

- **Description**: Build governed tool-use SFT/evaluation cases for selection, schema-conformant arguments, result handling, error recovery, sequential/parallel calls, and permission boundaries.

- **Dependencies**: `AGT-001`, `AGT-002`, `CHAT-002`

- **Relevant files**: `src/inference/generator.py`, `src/inference/local_tools.py`, `src/evaluation/benchmarks.py`, `tests/test_generation.py`, `tests/test_local_tools.py`

- **Validation**: `.venv/bin/pytest tests/test_tool_use_evaluation.py tests/test_local_tools.py tests/test_mcp_client.py tests/test_workspace_agent.py -q` (30 passed).

- **Acceptance criteria**: Versioned governed fixtures cover selection, schema-conformant arguments, result handling, error recovery, sequential calls, parallel-call intent, and permission denial. Runtime execution remains bounded and sequential; parallel intent is not execution authorization.

### EVAL-002: Comprehensive Capability Regression Matrix

- **ID**: `EVAL-002`

- **Status**: `COMPLETED`

- **Priority**: `P0`

- **Description**: Version a release-gating evaluation matrix for knowledge, math, code, reasoning, instruction following, structured JSON, tools, RAG, long context, safety, hallucination, and refusal behavior.

- **Dependencies**: `RSN-003`, `SAFE-001`, `EVAL-001`, `CHAT-002`, `AGT-003`, `EMB-001`

- **Relevant files**: `src/evaluation/benchmarks.py`, `scripts/evaluate_benchmarks.py`, `configs/evaluation.domains.yaml`, `tests/test_evaluation_regressions.py`

- **Validation**: `.venv/bin/pytest tests/test_release_matrix.py tests/test_evaluation_regressions.py tests/test_prompt_safety.py tests/test_tool_use_evaluation.py tests/test_rag.py tests/test_serving.py tests/test_positional.py -q` (102 passed).

- **Acceptance criteria**: A versioned mandatory release matrix maps every required capability category to an existing deterministic artifact and test command. It is a regression-contract gate, not a claim of comprehensive checkpoint capability.

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

- **Status**: `COMPLETED`

- **Priority**: `P2`

- **Description**: Add configuration-driven domain curricula, per-domain loss reporting, activation/gradient diagnostics, and safe schedule resume semantics.

- **Dependencies**: `DATA-002`

- **Relevant files**: `src/datasets/sampler.py`, `src/training/trainer.py`, `configs/pretraining.gpu.yaml`, `tests/test_training_data.py`

- **Validation**: `.venv/bin/pytest tests/test_training_data.py tests/test_training_system.py -q` (58 passed)

- **Acceptance criteria**: Epoch-indexed curricula validate a stable source ordering and update grouped sampler weights deterministically; sampler/trainer state preserves the active stage for resume; history records gradient, clipping, and output-magnitude diagnostics alongside existing per-domain validation loss.

### EMB-001: Dedicated Embedding and Retrieval Quality Evaluation

- **ID**: `EMB-001`

- **Status**: `TODO`

- **Priority**: `P2`

- **Description**: Define a dedicated embedding-model interface and retrieval recall/reranking evaluation rather than relying solely on lexical or generic model representations.

- **Dependencies**: `RAG-002`

- **Relevant files**: `src/inference/rag.py`, `src/evaluation/benchmarks.py`, `tests/test_rag.py`

### QNT-002: Interoperable Quantized Deployment Formats

- **ID**: `QNT-002`

- **Status**: `TODO`

- **Priority**: `P2`

- **Description**: Evaluate and implement verified GPTQ/AWQ/GGUF-compatible export or import paths, including calibration provenance, architecture compatibility checks, and quality/latency/memory regression measurements.

- **Dependencies**: `QNT-001`, `REG-001`

- **Relevant files**: `src/inference/quantization.py`, `scripts/export.py`, `tests/test_inference_precision.py`, `tests/test_export.py`

### AGT-004: Stateful Agent Runtime and Human Approval

- **ID**: `AGT-004`

- **Status**: `TODO`

- **Priority**: `P2`

- **Description**: Add a bounded agent state machine for planning, observations, task decomposition, explicit human approval gates, and recoverable multi-step execution without granting unbounded tool authority.

- **Dependencies**: `AGT-002`, `AGT-003`, `MEM-001`

- **Relevant files**: `src/inference/generator.py`, `src/inference/local_tools.py`, `src/mcp/client.py`, `tests/test_generation.py`, `tests/test_mcp_client.py`

### MEM-002: Long-Term Semantic and Episodic Memory

- **ID**: `MEM-002`

- **Status**: `TODO`

- **Priority**: `P3`

- **Description**: Extend session-only memory with opt-in semantic and episodic retrieval, explicit retention policies, user-scoped deletion, and evaluation of memory relevance and privacy boundaries.

- **Dependencies**: `MEM-001`, `EMB-001`

- **Relevant files**: `src/inference/chat_session.py`, `src/inference/rag.py`, `src/serving/api.py`, `tests/test_chat_session.py`, `tests/test_rag.py`

### ATT-002: Advanced Long-Context Attention Research Profiles

- **ID**: `ATT-002`

- **Status**: `TODO`

- **Priority**: `P3`

- **Description**: Prototype and benchmark sliding-window, sparse, linear, or hybrid attention profiles against the dense GQA baseline; retain deterministic fallback and checkpoint compatibility guarantees.

- **Dependencies**: `ATT-001`, `CTX-002`

- **Relevant files**: `src/model/attention.py`, `src/model/config.py`, `tests/test_attention.py`

### SCALE-004: QLoRA Fine-Tuning Profile

- **ID**: `SCALE-004`

- **Status**: `TODO`

- **Priority**: `P2`

- **Description**: Add a calibrated low-bit base-model plus LoRA fine-tuning profile suitable for constrained GPUs, with merge/export compatibility and quality regression checks.

- **Dependencies**: `SCALE-003`, `QNT-002`

- **Relevant files**: `src/training/peft.py`, `src/inference/quantization.py`, `configs/finetuning.gpu.yaml`, `tests/test_peft.py`, `tests/test_inference_precision.py`

### SCALE-005: Tensor, Pipeline, and Expert Parallelism Readiness

- **ID**: `SCALE-005`

- **Status**: `TODO`

- **Priority**: `P3`

- **Description**: Define and validate topology-aware tensor, pipeline, and MoE expert-parallel execution/checkpoint contracts before claiming multi-node large-model support.

- **Dependencies**: `SCALE-001`, `SCALE-002`

- **Relevant files**: `src/inference/tensor_parallel.py`, `src/training/multinode.py`, `src/training/distributed_checkpoint.py`, `tests/test_tensor_parallel.py`, `tests/test_multinode.py`

### VIS-003: Audio and Video Multimodal Research Profile

- **ID**: `VIS-003`

- **Status**: `TODO`

- **Priority**: `P3`

- **Description**: Establish separately versioned audio/video data, encoder, projection, safety, and evaluation contracts; do not extend the image projector implicitly into unsupported modalities.

- **Dependencies**: `VIS-002`, `DATA-003`

- **Relevant files**: `src/vision/encoder.py`, `src/multimodal/model.py`, `tests/test_vision_models.py`

### DOC-001: Evidence-Backed Capability-State Reconciliation

- **ID**: `DOC-001`

- **Status**: `COMPLETED`

- **Priority**: `P0`

- **Description**: Reconcile `README.md`, `required.md`, `docs/CURRENT_STATE.md`, `docs/TARGET_STATE.md`, and `docs/TASKS.md` against authoritative source/tests. Classify each capability as implemented, partial, planned, or research and resolve contradictory agent/tool-orchestration claims.

- **Dependencies**: `OPS-002`

- **Relevant files**: `README.md`, `required.md`, `docs/CURRENT_STATE.md`, `docs/TARGET_STATE.md`, `docs/TASKS.md`, `docs/CHANGELOG.md`

- **Acceptance criteria**: Every state claim links to an authoritative implementation/test or a clearly scoped external-validation requirement; no two current-state documents assign conflicting statuses to the same capability.

- **Validation**: Reconciled active context, tool-orchestration, memory, and release-gate claims against `src/serving/backend.py`, `src/inference/local_tools.py`, `src/inference/chat_session.py`, and their regression tests. Hardware, corpus, and trained-checkpoint claims remain explicitly external-validation requirements.

### API-001: Session Context-Window Inspection and Compaction

- **ID**: `API-001`

- **Status**: `COMPLETED`

- **Priority**: `P2`

- **Description**: Add authenticated, opt-in session context usage metrics and an explicit compaction operation for API clients.

- **Dependencies**: `MEM-001`

- **Relevant files**: `src/serving/api.py`, `src/inference/context.py`, `tests/test_serving.py`

- **Acceptance criteria**: Authenticated opt-in clients can inspect
  tokenizer-measured stored-session capacity and explicitly compact history to
  the requested response reserve. Request-scoped tool definitions, files, and
  tool results remain unpersisted and are reported as such.

- **Validation**: `./.venv/bin/python -m pytest tests/test_serving.py -q`
  — 42 passed.

### AGT-005: Agent Task Success Benchmark

- **ID**: `AGT-005`

- **Track**: `AGENT`

- **Status**: `TODO`

- **Priority**: `P1`

- **Description**: Measure end-to-end agent task success, tool selection, argument validity, recovery, safety-boundary compliance, and execution efficiency with versioned deterministic fixtures.

- **Dependencies**: `AGT-003`, `AGT-004`, `EVAL-002`

- **Relevant files**: `src/mcp/orchestration.py`, `src/inference/local_tools.py`, `src/evaluation/benchmarks.py`, `tests/test_mcp_client.py`, `tests/test_local_tools.py`

---

## Sheet-Sync Additions — 2026-09-20

The following tasks are present in `Gopi_LLM_Final_Task_Sheet.xlsx` but were not present in this task-list file. They are appended here without deleting or silently rewriting existing task definitions.

### DATA-004: Dataset Contamination Audit

- **ID**: `DATA-004`

- **Track**: `DATA`

- **Status**: `TODO`

- **Priority**: `P0`

- **Description**: Independent train/eval contamination and near-duplicate audit

---

### DATA-005: Dataset Quality Scoring

- **ID**: `DATA-005`

- **Track**: `DATA`

- **Status**: `TODO`

- **Priority**: `P0`

- **Description**: Per-document quality scoring and source weighting

---

### DATA-006: Data Mixture Ablation Framework

- **ID**: `DATA-006`

- **Track**: `DATA`

- **Status**: `TODO`

- **Priority**: `P0`

- **Description**: Compare corpus mixtures against capability metrics

---

### DATA-007: Token/Sequence Packing Efficiency

- **ID**: `DATA-007`

- **Track**: `DATA`

- **Status**: `TODO`

- **Priority**: `P1`

- **Description**: Measure useful-token vs padding/packing waste

---

### TRAIN-001: Scaling-Law Experiment Framework

- **ID**: `TRAIN-001`

- **Track**: `TRAINING`

- **Status**: `COMPLETED`

- **Priority**: `P0`

- **Description**: Parameters/tokens/compute vs validation and capability scaling

- **Relevant files**: `src/training/scaling_laws.py`, `tests/test_scaling_laws.py`

- **Validation**: `.venv/bin/pytest tests/test_scaling_laws.py tests/test_training_accounting.py -q` (7 passed).

- **Acceptance criteria**: Versioned observations validate parameter/token/loss/capability values and deterministically report log-compute trend slopes. Real scaling conclusions require recorded experiments.

---

### TRAIN-002: Compute and Token Budget Accounting

- **ID**: `TRAIN-002`

- **Track**: `TRAINING`

- **Status**: `COMPLETED`

- **Priority**: `P0`

- **Description**: Reproducible tokens, FLOPs, GPU-hours and throughput accounting

- **Relevant files**: `src/training/accounting.py`, `src/training/trainer.py`, `tests/test_training_accounting.py`

- **Validation**: `.venv/bin/pytest tests/test_training_accounting.py tests/test_training_system.py -q` (48 passed).

- **Acceptance criteria**: Trainer counters produce a reproducible report for supervised tokens, optimizer steps, elapsed device-hours, tokens/second, and the explicit `6 × parameters × tokens` FLOP estimate. It does not represent a hardware FLOP measurement.

---

### TRAIN-003: Hyperparameter Sweep Framework

- **ID**: `TRAIN-003`

- **Track**: `TRAINING`

- **Status**: `COMPLETED`

- **Priority**: `P1`

- **Description**: Systematic LR/batch/scheduler/optimizer experiment runner

- **Relevant files**: `src/training/sweeps.py`, `scripts/run_sweep.py`, `tests/test_training_sweeps.py`

- **Validation**: `.venv/bin/pytest tests/test_training_sweeps.py tests/test_training_system.py tests/test_optim.py -q`

- **Acceptance criteria**: Deterministic versioned manifests expand validated Cartesian trial grids. The CLI plans by default and only executes sequential training trials with `--execute`.

---

### TRAIN-004: Training Stability Diagnostics

- **ID**: `TRAIN-004`

- **Track**: `TRAINING`

- **Status**: `COMPLETED`

- **Priority**: `P1`

- **Description**: Gradient/activation/update/loss stability monitoring

- **Relevant files**: `src/training/trainer.py`, `tests/test_training_system.py`

- **Validation**: `.venv/bin/pytest tests/test_training_system.py tests/test_optim.py tests/test_loss.py -q`

- **Acceptance criteria**: Persisted trainer and epoch-history diagnostics record loss, gradient norms, logit activation scale, parameter scale, and gradient-to-parameter update-scale proxy without retaining activation or parameter snapshots.

---

### TRAIN-005: Checkpoint Selection and Model Promotion

- **ID**: `TRAIN-005`

- **Track**: `TRAINING`

- **Status**: `TODO`

- **Priority**: `P1`

- **Description**: Multi-axis checkpoint promotion instead of loss-only selection

---

### CHAT-003: Instruction Dataset Quality Pipeline

- **ID**: `CHAT-003`

- **Track**: `CHAT`

- **Status**: `TODO`

- **Priority**: `P0`

- **Description**: Quality filtering, balancing, verification and category coverage

---

### ALIGN-001: Preference Dataset Pipeline

- **ID**: `ALIGN-001`

- **Track**: `ALIGNMENT`

- **Status**: `TODO`

- **Priority**: `P0`

- **Description**: Chosen/rejected pair quality, bias and provenance checks

---

### ALIGN-002: DPO Evaluation and Regression

- **ID**: `ALIGN-002`

- **Track**: `ALIGNMENT`

- **Status**: `TODO`

- **Priority**: `P1`

- **Description**: SFT→DPO capability/regression comparison

---

### ALIGN-003: Alignment Method Experiment Interface

- **ID**: `ALIGN-003`

- **Track**: `ALIGNMENT`

- **Status**: `TODO`

- **Priority**: `P2`

- **Description**: Controlled comparison of DPO/IPO/ORPO/KTO/etc.

---

### RSN-004: Reasoning Verification Pipeline

- **ID**: `RSN-004`

- **Track**: `REASONING`

- **Status**: `TODO`

- **Priority**: `P0`

- **Description**: Programmatic/math/code verification of reasoning outputs

---

### RSN-005: Reasoning Difficulty Curriculum

- **ID**: `RSN-005`

- **Track**: `REASONING`

- **Status**: `TODO`

- **Priority**: `P1`

- **Description**: Difficulty-ordered reasoning training

---

### EVAL-003: External Benchmark Adapter Framework

- **ID**: `EVAL-003`

- **Track**: `EVALUATION`

- **Status**: `TODO`

- **Priority**: `P0`

- **Description**: Versioned benchmark adapters with reproducible scoring controls

---

### EVAL-004: Dynamic / Anti-Contamination Evaluation

- **ID**: `EVAL-004`

- **Track**: `EVALUATION`

- **Status**: `TODO`

- **Priority**: `P1`

- **Description**: Generated math/logic/instruction/tool tests

---

### EVAL-005: Calibration and Confidence Evaluation

- **ID**: `EVAL-005`

- **Track**: `EVALUATION`

- **Status**: `TODO`

- **Priority**: `P2`

- **Description**: Confidence-vs-correctness metrics

---

### EVAL-006: Hallucination and Uncertainty Evaluation

- **ID**: `EVAL-006`

- **Track**: `EVALUATION`

- **Status**: `TODO`

- **Priority**: `P1`

- **Description**: Unknown/false-premise/citation/RAG hallucination tests

---

### EVAL-007: Release Regression Gate

- **ID**: `EVAL-007`

- **Track**: `EVALUATION`

- **Status**: `TODO`

- **Priority**: `P0`

- **Description**: Automatic protected-capability regression blocking

---

### TOKEN-002: Tokenizer Quality Benchmark

- **ID**: `TOKEN-002`

- **Track**: `TOKENIZER`

- **Status**: `TODO`

- **Priority**: `P1`

- **Description**: Efficiency across languages/code/JSON/Unicode/numbers

---

### TOKEN-003: Tokenizer Version Compatibility Gate

- **ID**: `TOKEN-003`

- **Track**: `TOKENIZER`

- **Status**: `TODO`

- **Priority**: `P1`

- **Description**: ID/special-token/checkpoint compatibility across releases

---

### CTX-003: Context Scaling Ablation

- **ID**: `CTX-003`

- **Track**: `CONTEXT`

- **Status**: `TODO`

- **Priority**: `P1`

- **Description**: 512→1K→2K→4K→8K quality/memory/throughput study

---

### INF-003: End-to-End Throughput Benchmark

- **ID**: `INF-003`

- **Track**: `INFERENCE`

- **Status**: `TODO`

- **Priority**: `P1`

- **Description**: TTFT/ITL/TPS/throughput/memory across batch and precision

---

### INF-004: Serving Load and Soak Testing

- **ID**: `INF-004`

- **Track**: `INFERENCE`

- **Status**: `TODO`

- **Priority**: `P1`

- **Description**: 10m/1h/6h/24h sustained load and leak detection

---

### INF-006: Cancellation and Failure Stress Testing

- **ID**: `INF-006`

- **Track**: `INFERENCE`

- **Status**: `TODO`

- **Priority**: `P1`

- **Description**: Disconnect/timeout/OOM/tool/server-restart cleanup tests

---

### API-002: Structured Output / JSON Schema Enforcement

* **ID**: `API-002`

* **Title**: Structured Output / JSON Schema Enforcement

* **Status**: `TODO`

* **Priority**: `P0`

* **Description**: Implement strict JSON Schema-based structured outputs for Gopi LLM API responses.

* **Why**: Applications and agents need predictable, schema-conforming responses instead of merely valid JSON.

* **Dependencies**: `API-001`

* **Relevant files**:

  * `src/api/chat_completions.*`
  * `src/runtime/generation.*`
  * `src/schema/*`
  * `tests/api/structured_outputs.*`

* **Implementation notes**:

  * Accept `response_format.type = json_schema`.
  * Accept schema name and JSON Schema definition.
  * Validate schema compatibility.
  * Constrain or validate generated output.
  * Support `strict: true`.
  * Return refusal/incomplete states.
  * Handle malformed schemas and incomplete generation.

* **Validation**: Test valid schemas, invalid schemas, nested objects, arrays, required fields, additional properties, malformed generation, and incomplete generation.

* **Acceptance criteria**: Every successful structured-output request returns data conforming to the supplied JSON Schema.

---

### API-003: Responses API

* **ID**: `API-003`

* **Title**: OpenAI-Compatible Responses API

* **Status**: `TODO`

* **Priority**: `P1`

* **Description**: Implement `/v1/responses` as a richer agent/event API while maintaining `/v1/chat/completions` compatibility.

* **Why**: Provides a unified API for model generation, tools, RAG, and future agent workflows.

* **Dependencies**: `API-001`, `AGT-001`

* **Relevant files**:

  * `src/api/responses.*`
  * `src/api/chat_completions.*`
  * `src/runtime/agent.*`
  * `tests/api/responses.*`

* **Implementation notes**:

  * Add `POST /v1/responses`.
  * Convert requests into a unified internal request representation.
  * Integrate model, tools, and RAG.
  * Support response/event streaming.
  * Preserve Chat Completions compatibility.

* **Validation**: Compare equivalent Chat Completions and Responses requests and verify consistent model behavior.

* **Acceptance criteria**: `/v1/responses` successfully handles normal generation, streaming, tool interactions, and RAG workflows.

---

### RSN-007: Configurable Reasoning Budget

* **ID**: `RSN-007`

* **Title**: Configurable Reasoning Budget

* **Status**: `TODO`

* **Priority**: `P1`

* **Description**: Add configurable reasoning effort controls to the model runtime.

* **Why**: Allows applications to trade off reasoning depth, latency, and compute usage.

* **Dependencies**: `TRAIN-005`, `INF-001`

* **Relevant files**:

  * `src/runtime/reasoning.*`
  * `src/inference/generation.*`
  * `src/api/request_schema.*`
  * `tests/reasoning/*`

* **Implementation notes**:

  * Support `none`, `low`, `medium`, and `high`.
  * Map reasoning effort to an actual runtime budget.
  * Track reasoning token usage.
  * Do not expose the capability until runtime behavior is implemented and measured.

* **Validation**: Run the same benchmark across reasoning levels and measure quality, latency, and token usage.

* **Acceptance criteria**: Each supported reasoning level produces measurable and reproducible runtime behavior.

---

### AGT-006: Native MCP Tool Execution Contract

* **ID**: `AGT-006`

* **Title**: Native MCP Tool Execution Contract

* **Status**: `TODO`

* **Priority**: `P1`

* **Description**: Implement MCP as a first-class tool execution mechanism.

* **Why**: Allows Gopi LLM to interact with external MCP servers using the same security and execution policies as normal tools.

* **Dependencies**: `AGT-001`, `SEC-*`

* **Relevant files**:

  * `src/agents/mcp.*`
  * `src/tools/*`
  * `src/security/authorization.*`
  * `tests/mcp/*`

* **Implementation notes**:

  * Support MCP server configuration.
  * Support `server_label`.
  * Support `server_url`.
  * Support approval policies.
  * Apply authorization, timeout, cancellation, audit, and approval policies.
  * Validate MCP tool results.

* **Validation**: Test successful MCP calls, rejected calls, timeout, cancellation, invalid results, and approval-required flows.

* **Acceptance criteria**: MCP tools execute through the same controlled tool lifecycle as native tools.

---

### RAG-005: Cross-Encoder / Reranker Interface

* **ID**: `RAG-005`

* **Title**: Cross-Encoder / Reranker Interface

* **Status**: `TODO`

* **Priority**: `P1`

* **Description**: Add a reranking layer between retrieval and LLM generation.

* **Why**: Improve the relevance of retrieved documents before sending context to Gopi LLM.

* **Dependencies**: `RAG-003`, `API-010`

* **Relevant files**:

  * `src/rag/retriever.*`
  * `src/rag/reranker.*`
  * `src/rag/pipeline.*`
  * `tests/rag/reranker.*`

* **Implementation notes**:

  * Retrieve an initial candidate set.
  * Rerank candidates using a cross-encoder/reranker.
  * Return top 5–10 documents.
  * Preserve document metadata and citations.
  * Measure retrieval recall and reranking quality.

* **Validation**: Evaluate retrieval recall, reranking quality, answer faithfulness, and citation correctness.

* **Acceptance criteria**: RAG pipeline produces measurable improvements or validated ranking quality before generation.

---

### API-004: Token Usage & Cache Accounting Contract

* **ID**: `API-004`

* **Title**: Token Usage & Cache Accounting Contract

* **Status**: `TODO`

* **Priority**: `P1`

* **Description**: Implement standardized token and cache usage reporting for every completion.

* **Why**: Required for billing, benchmarking, cache analysis, capacity planning, and inference reports.

* **Dependencies**: `INF-001`

* **Relevant files**:

  * `src/api/usage.*`
  * `src/inference/tokenizer.*`
  * `src/cache/*`
  * `tests/api/usage.*`

* **Implementation notes**:

  * Report `prompt_tokens`.
  * Report `completion_tokens`.
  * Report `total_tokens`.
  * Track `cached_tokens`.
  * Track `reasoning_tokens` when implemented.
  * Track prefix-cache hits and misses.

* **Validation**: Compare reported token counts against tokenizer-generated counts and cache events.

* **Acceptance criteria**: API usage statistics are accurate, consistent, and available in every completion response.

---

### API-005: Generation Parameter Compatibility Suite

* **ID**: `API-005`

* **Title**: Generation Parameter Compatibility Suite

* **Status**: `TODO`

* **Priority**: `P1`

* **Description**: Implement and test OpenAI-compatible generation controls supported by the actual sampler.

* **Why**: Clients should only receive parameters that the Gopi LLM runtime genuinely supports.

* **Dependencies**: `INF-002`

* **Relevant files**:

  * `src/inference/sampler.*`
  * `src/api/request_schema.*`
  * `tests/api/generation_parameters.*`

* **Implementation notes**:

  * Support `temperature`.
  * Support `top_p`.
  * Support `top_k`.
  * Support `min_p`.
  * Support `max_tokens`.
  * Support `stop`.
  * Support `seed`.
  * Support repetition/presence/frequency penalties where implemented.
  * Validate unsupported parameters.

* **Validation**: Run deterministic generation tests with fixed seeds and parameter-specific sampler tests.

* **Acceptance criteria**: Every exposed parameter changes runtime behavior according to its documented contract.

---

### API-006: Request Cancellation Contract

* **ID**: `API-006`

* **Title**: Request Cancellation Contract

* **Status**: `TODO`

* **Priority**: `P0`

* **Description**: Implement cancellation of active generation when clients disconnect or explicitly cancel requests.

* **Why**: Prevents wasted GPU computation and releases model/runtime resources promptly.

* **Dependencies**: `INF-006`

* **Relevant files**:

  * `src/api/server.*`
  * `src/inference/generation.*`
  * `src/runtime/cancellation.*`
  * `src/tools/executor.*`
  * `tests/inference/cancellation.*`

* **Implementation notes**:

  * Detect client disconnect.
  * Stop generation.
  * Release KV-cache resources.
  * Cancel active tools.
  * Release GPU/runtime resources.
  * Propagate cancellation through the complete request lifecycle.

* **Validation**: Disconnect clients during short and long generations and verify that generation and tool execution stop.

* **Acceptance criteria**: Cancelled requests stop consuming generation resources within the defined cancellation timeout.

---

### API-007: Model Capability Discovery Contract

* **ID**: `API-007`

* **Title**: Model Capability Discovery Contract

* **Status**: `TODO`

* **Priority**: `P1`

* **Description**: Expose accurate model capabilities and architecture metadata through the model configuration/API.

* **Why**: Clients need to know which capabilities are actually implemented and validated.

* **Dependencies**: `EVAL-014`

* **Relevant files**:

  * `src/api/models.*`
  * `src/config/model.*`
  * `src/runtime/capabilities.*`
  * `tests/api/models.*`

* **Implementation notes**:

  * Expose chat capability.
  * Expose streaming.
  * Expose tool calling.
  * Expose structured outputs.
  * Expose reasoning state.
  * Expose vision/audio state.
  * Expose RAG/MCP state.
  * Expose context limits.
  * Expose architecture metadata.
  * Expose inference features such as KV cache and prefix caching.

* **Validation**: Compare advertised capabilities against automated integration tests.

* **Acceptance criteria**: A capability is advertised as `true` only after implementation and validation. 

---

### API-008: Protected Model Lifecycle API

* **ID**: `API-008`

* **Title**: Protected Model Lifecycle API

* **Status**: `TODO`

* **Priority**: `P2`

* **Description**: Implement administrative APIs for loading, unloading, and reloading models.

* **Why**: Enables controlled model lifecycle management in production serving environments.

* **Dependencies**: `SEC-*`, `INF-*`

* **Relevant files**:

  * `src/api/admin/models.*`
  * `src/model/loader.*`
  * `src/model/lifecycle.*`
  * `tests/api/model_lifecycle.*`

* **Implementation notes**:

  * Add `POST /admin/models/load`.
  * Add `POST /admin/models/unload`.
  * Add `POST /admin/models/reload`.
  * Require strong authentication and authorization.
  * Record lifecycle audit events.
  * Never expose unrestricted lifecycle operations publicly.

* **Validation**: Test load/unload/reload under normal and concurrent request conditions.

* **Acceptance criteria**: Model lifecycle operations are functional and protected from unauthorized access. 

---

### EVAL-014: OpenAI-Compatible API Conformance Suite

* **ID**: `EVAL-014`

* **Title**: OpenAI-Compatible API Conformance Suite

* **Status**: `TODO`

* **Priority**: `P0`

* **Description**: Build automated tests verifying that Gopi LLM follows the defined OpenAI-compatible API contract.

* **Why**: API compatibility must be demonstrated through actual request/response behavior rather than metadata.

* **Dependencies**: `API-001`, `API-002`, `API-003`, `API-004`, `API-005`, `API-006`

* **Relevant files**:

  * `tests/conformance/*`
  * `tests/api/*`
  * `scripts/run_api_conformance.*`

* **Implementation notes**:

  * Test request validation.
  * Test response structure.
  * Test streaming.
  * Test usage reporting.
  * Test errors.
  * Test generation parameters.
  * Test cancellation.
  * Test authentication.
  * Test model discovery.

* **Validation**: Run the complete suite against every release candidate.

* **Acceptance criteria**: No P0 compatibility regression is allowed before release.

---

### EVAL-015: Tool Calling Conformance Suite

* **ID**: `EVAL-015`

* **Title**: Tool Calling Conformance Suite

* **Status**: `TODO`

* **Priority**: `P0`

* **Description**: Create a complete automated test suite for the tool-calling lifecycle.

* **Why**: Tool calling requires more than exposing `toolCalling: true`; validation, authorization, execution, errors, and cancellation must all work correctly. 

* **Dependencies**: `AGT-001`, `API-002`

* **Relevant files**:

  * `tests/tools/*`
  * `tests/agents/*`
  * `tests/conformance/tool_calling.*`

* **Implementation notes**:

  * Test `tool_choice=none`.
  * Test `tool_choice=auto`.
  * Test `tool_choice=required`.
  * Test specific function selection.
  * Test strict schemas.
  * Test malformed arguments.
  * Test authorization.
  * Test timeout.
  * Test cancellation.
  * Test result-size limits.
  * Test maximum tool iterations.
  * Test audit events.
  * Test approval flows.
  * Test parallel tool calls where supported.

* **Validation**: Execute the full lifecycle against mock and real tools.

* **Acceptance criteria**: Tool calls cannot bypass schema validation, authorization, timeout, cancellation, or audit controls.

---

### EVAL-016: Structured Output Conformance Suite

* **ID**: `EVAL-016`

* **Title**: Structured Output Conformance Suite

* **Status**: `TODO`

* **Priority**: `P0`

* **Description**: Build automated conformance tests for JSON Schema structured outputs.

* **Why**: Structured Outputs are an API-level guarantee and must be validated independently from ordinary JSON generation.

* **Dependencies**: `API-002`

* **Relevant files**:

  * `tests/structured_outputs/*`
  * `tests/conformance/structured_outputs.*`

* **Implementation notes**:

  * Test primitive fields.
  * Test nested objects.
  * Test arrays.
  * Test required properties.
  * Test `additionalProperties=false`.
  * Test strict mode.
  * Test invalid schemas.
  * Test incomplete generation.
  * Test refusal states.
  * Test streaming structured output.

* **Validation**: Validate every successful response against the original JSON Schema.

* **Acceptance criteria**: 100% of successful structured-output test cases conform to their supplied schemas.

---

## Additional platform tasks from the same plan

The document also identifies several capabilities that should become implementation tasks even where a separate task ID is not explicitly assigned in the recommended task table. The overall capability set includes chat completions, streaming, tool calling, structured outputs, reasoning controls, Responses API, MCP, sessions/context, embeddings, RAG, reranking, usage accounting, deterministic controls, cancellation, prefix caching, lifecycle management, evaluation, and release/regression gates. 

### EMB-001: Embeddings API

* **ID**: `EMB-001`

* **Title**: Gopi Embeddings API

* **Status**: `TODO`

* **Priority**: `P1`

* **Description**: Implement `POST /v1/embeddings` using a dedicated embedding model.

* **Why**: Provides a standardized embedding service for the Gopi RAG platform.

* **Dependencies**: `MODEL-EMB-001`

* **Relevant files**:

  * `src/api/embeddings.*`
  * `src/embeddings/*`
  * `tests/api/embeddings.*`

* **Implementation notes**:

  * Support single and batch text inputs.
  * Return embedding vectors.
  * Return token usage.
  * Expose embedding model metadata.
  * Do not assume the generative Gopi model is automatically an optimal embedding model.

* **Validation**: Test dimensions, deterministic behavior, batching, malformed input, and similarity quality.

* **Acceptance criteria**: `/v1/embeddings` returns valid embeddings with documented dimensions and usage metadata. 

---

### INF-007: Prefix Cache Observability

* **ID**: `INF-007`

* **Title**: Prefix / Prompt Cache Observability

* **Status**: `TODO`

* **Priority**: `P1`

* **Description**: Expose measurable prefix-cache behavior for repeated system prompts, tool definitions, and RAG context.

* **Why**: Cached prefixes can reduce repeated prefill work and should be measurable.

* **Dependencies**: `INF-003`

* **Relevant files**:

  * `src/cache/prefix.*`
  * `src/inference/metrics.*`
  * `src/api/usage.*`

* **Implementation notes**:

  * Track cache hits.
  * Track cache misses.
  * Track cached tokens.
  * Track prefill saved.
  * Track cache memory.
  * Track cache evictions.

* **Validation**: Run repeated requests with identical prefixes and verify cache-hit metrics.

* **Acceptance criteria**: Cache behavior is visible through usage/metrics and can be evaluated quantitatively. 

---

### OPS-001: Model Health & Readiness API

* **ID**: `OPS-001`

* **Title**: Model Health, Readiness & Metrics Endpoints

* **Status**: `TODO`

* **Priority**: `P1`

* **Description**: Provide operational endpoints for model health, readiness, discovery, and metrics.

* **Why**: Production deployments require infrastructure-level health and observability.

* **Dependencies**: `API-001`

* **Relevant files**:

  * `src/api/health.*`
  * `src/api/models.*`
  * `src/api/metrics.*`
  * `tests/api/health.*`

* **Implementation notes**:

  * `GET /v1/models`
  * `GET /health`
  * `GET /ready`
  * `GET /metrics`

* **Validation**: Test healthy, loading, unavailable, degraded, and model-reloading states.

* **Acceptance criteria**: Deployment infrastructure can reliably determine whether Gopi LLM is alive, ready, and serving. 

---

### EVAL-017: Evaluation & Release Regression Pipeline

* **ID**: `EVAL-017`

* **Title**: Evaluation, Regression & Checkpoint Promotion Pipeline

* **Status**: `TODO`

* **Priority**: `P0`

* **Description**: Connect training checkpoints to evaluation, regression comparison, promotion, and release-candidate generation.

* **Why**: Model capabilities should only become publicly advertised after passing implementation, quality, and performance validation.

* **Dependencies**: `TRAIN-005`, `EVAL-007`, `RELEASE-001`

* **Relevant files**:

  * `scripts/eval.*`
  * `src/evaluation/*`
  * `src/release/*`
  * `tests/evaluation/*`

* **Implementation notes**:

  * Evaluate every candidate checkpoint.
  * Compare against previous checkpoint.
  * Run API conformance.
  * Run tool-calling conformance.
  * Run structured-output conformance.
  * Run quality benchmarks.
  * Run performance benchmarks.
  * Promote only passing checkpoints.

* **Validation**: Execute the complete pipeline from checkpoint → evaluation → regression → promotion → release candidate.

* **Acceptance criteria**: A model capability is not marked `true` in metadata until its required evaluation and performance gates pass. 

---

### RAG-003: Retrieval Recall and Faithfulness Benchmark

- **ID**: `RAG-003`

- **Track**: `RAG`

- **Status**: `TODO`

- **Priority**: `P1`

- **Description**: Recall, reranking and answer/citation faithfulness

---

### RELEASE-001: Model Release Candidate Pipeline

- **ID**: `RELEASE-001`

- **Track**: `RELEASE`

- **Status**: `TODO`

- **Priority**: `P0`

- **Description**: Training→eval→safety→regression→manifest→RC

---

### RELEASE-002: Model Card and Dataset Card

- **ID**: `RELEASE-002`

- **Track**: `RELEASE`

- **Status**: `TODO`

- **Priority**: `P1`

- **Description**: Release documentation, limitations, provenance and evaluation

---

### RELEASE-003: Reproducible Model Recreation

- **ID**: `RELEASE-003`

- **Track**: `RELEASE`

- **Status**: `TODO`

- **Priority**: `P1`

- **Description**: Recreate experiment from commit/config/data/tokenizer manifest

---

## Task List Synchronization Audit — 2026-09-20

- Original task-list IDs: **48**
- Sheet task IDs: **72**
- IDs present in both: **42**
- Sheet-only IDs added above: **30**
- Same-ID agent benchmark mapped to distinct `AGT-005`: **1**
- Task-list-only IDs retained: **6**

### Task-list-only IDs retained

`QNT-002`, `MEM-002`, `ATT-002`, `SCALE-004`, `SCALE-005`, `VIS-003`

### Same-ID definition/status reconciliation

- `AGT-004` remains **Stateful Agent Runtime and Human Approval** / `TODO`; the sheet's distinct **Agent Task Success Benchmark** is tracked as `AGT-005`.
- `CHAT-001` — task list: **Chat Template Standardization & Loss Masking** / `COMPLETED`; sheet: **Standardized Chat Templating with Prompt Loss Masking** / `COMPLETED`.
- `CUR-001` is **COMPLETED** locally with targeted validation; the sheet's `TODO` status is stale.
- `DATA-001` — task list: **MinHash Deduplication Pipeline** / `COMPLETED`; sheet: **Scalable MinHash LSH Corpus Deduplication** / `COMPLETED`.
- `DOC-001` incorporates the sheet's documentation consistency gate and is prioritized `P0`.
- `OPS-002` — task list: **AI Agent Documentation & Knowledge System** / `COMPLETED`; sheet: **AI Agent Documentation & Engineering Knowledge System** / `COMPLETED`.
- `RSN-001` — task list: **Chain-of-Thought Reasoning Pipeline** / `COMPLETED`; sheet: **Structured Chain-of-Thought Reasoning Pipeline** / `COMPLETED`.
