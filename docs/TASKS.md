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
- **Status**: `IN_PROGRESS`
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
- **Status**: `TODO`
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
- **Status**: `TODO`
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
- **Status**: `TODO`
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
- **Status**: `TODO`
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
- **Status**: `TODO`
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
- **Status**: `TODO`
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
- **Status**: `TODO`
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
- **Status**: `TODO`
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
- **Status**: `TODO`
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
- **Status**: `TODO`
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

