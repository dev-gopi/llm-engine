# Changelog (`docs/CHANGELOG.md`)

Notable user-visible changes and architectural milestones are recorded here.

---

## [Unreleased] — 2026-09-19: AI Agent Optimization, Audit & Engineering Knowledge System

### Added
- **MEM-001 Privacy-Bounded Session Memory**: Added deterministic, session-only
  memory retrieval and explicitly enabled, API-key-protected endpoints to read
  or delete persisted session memory. The interface is disabled by default.
- **QNT-001 Portable Low-Precision Artifacts**: Added FP16/BF16 export
  selection, self-describing packed INT4 safetensors artifacts, and opt-in
  per-token scaled INT8 paged-KV storage with explicit allocated-byte
  accounting.
- **SAFE-001 Safety Regression Probes**: Added a versioned deterministic
  guardrail manifest for prompt injection, harmful requests, and benign
  controls, with privacy-preserving aggregate results.
- **OBS-001 Serving Observability**: Added bounded p50/p95 latency, p50
  time-to-first-token, queue delay, token throughput, error-rate, and optional
  paged-KV utilization metrics.
- **RSN-003 Reasoning and Code Evaluation**: Added a versioned reasoning/code
  JSONL manifest and ensured benchmark loading preserves thinking-trace and
  answer-length scoring controls.
- **RAG-002 Reranking and Citations**: Added deterministic lexical-coverage
  reranking, a total retrieval-context budget, and source URLs alongside
  numeric RAG citations.
- **SEC-001 Auditable Serving Operations**: Added bounded, authenticated audit
  events for protected API operations. Events deliberately exclude prompts,
  credentials, tool arguments, and tool results.
- **Capability Backlog Audit**: Added dedicated tasks for efficient-attention
  selection, safety evaluation, serving observability, recovery drills,
  curriculum diagnostics, and embedding/retrieval-quality evaluation.
- **TKN-001 Agent Protocol Token Extension**: Added a checkpoint-compatible,
  append-only special-token extension for system, user, assistant, tool,
  thinking, and end delimiters; existing vocabulary IDs remain unchanged.
- **AGT-002 Multi-Step MCP Orchestration**: Added bounded sequential tool
  planning and execution with server/tool allowlists, per-tool argument schema
  validation, untrusted result injection, and structured recovery from errors.
- **INF-005 Paged-Attention Decode**: Active streamed decode now consumes
  per-layer page tables directly, combining page-level attention scores without
  materializing a contiguous request KV cache; numerical parity is tested
  against ordinary cached attention.
- **INF-002 Continuous Serving Validation**: Confirmed the existing token-step
  scheduler multiplexes active streams and admits queued work; split full
  page-table-aware attention execution into follow-up `INF-005`.
- **EVAL-001 Long-Context Evaluation**: Added deterministic 2K and 4K
  needle-in-a-haystack passkey probes with explicit placement and exact-answer
  scoring for checkpoint comparisons.
- **SCALE-003 LoRA Serving and Merging**: Added adapter-only state swapping,
  safe prefix-cache invalidation on adapter changes, standalone LoRA weight
  merging, and `--merge-lora` export support.
- **Documentation Capability Audit**: Updated `required.md`, current state,
  and target state to distinguish implemented capabilities from optional,
  partial, and future work based on the checked-in source and tests.
- **CTX-001 Long-Context RoPE Policies**: Added opt-in NTK-aware and YaRN
  rotary-frequency scaling, validated at 4K positions while keeping default
  checkpoint behavior unchanged.
- **AGT-001 Validated Tool Calls**: Added strict tool-call envelopes, JSON
  schema validation for local and MCP arguments, and a generator API that
  returns only validated tool-call payloads.
- **INF-001 Paged Prefix Cache Regression Contract**: Added verification that
  repeated paged-cache prompts reuse their immutable prefix state, skip a
  second prefill, and retain identical generation output.
- **RSN-001 Structured Reasoning Traces**: Added append-only thinking-token
  extensions, SFT validation and audit metrics for complete reasoning traces,
  and GSM8K-style scoring that requires a trace before the final answer.
- **CHAT-001 Canonical Chat SFT Format**: Added standard system, user, and
  assistant turn delimiters plus token-level SFT labels that ignore all prompt
  tokens and supervise assistant responses only.
- **DATA-001 MinHash LSH Deduplication**: Corpus cleaning now identifies near
  duplicates with five-token shingles, 128 deterministic MinHash permutations,
  and 32 collision bands; existing audit output reports the removals.
- **VIS-001 Vision Projector Training Invariant**: Frozen vision and language
  backbones now enter evaluation mode when the vision-language wrapper is
  constructed, ensuring deterministic projector-training targets from the
  first forward pass.
- **AI Agent Entry Point (`AGENTS.md`)**: Progressive context protocol directing AI coding agents to relevant files without whole-repository scanning.
- **Machine-Friendly Project Index (`docs/PROJECT_INDEX.md`)**: Instant lookup matrix mapping components to source code, configs, tests, and documentation.
- **Current-State Snapshot (`docs/CURRENT_STATE.md`)**: Verified architecture parameters (81.3M / 82.3M params, 16 layers, hidden 512, 8 query heads, 2 KV heads, GQA, RoPE, RMSNorm, SwiGLU, 4 GB VRAM budget).
- **Target-State Architecture (`docs/TARGET_STATE.md`)**: Clear separation of CURRENT, TARGET, and MIGRATION for long-context, hybrid attention, reasoning, and agent tool execution.
- **Phased Upgrade Roadmap (`docs/UPGRADE_ROADMAP.md`)**: 10-phase milestone plan with priorities, risks, dependencies, and validation criteria.
- **Agent Task System (`docs/TASKS.md`)**: Stable task IDs (`OPS-001`, `DATA-001`, `CHAT-001`, `RSN-001`, `INF-001`, `AGT-001`, `CTX-001`) with clear acceptance criteria.
- **Subsystem Technical Guides**: `docs/MODEL.md`, `docs/TOKENIZER.md`, `docs/DATASETS.md`, `docs/TRAINING.md`, `docs/EVALUATION.md`, `docs/INFERENCE.md`, `docs/SERVING.md`, `docs/AGENT_ARCHITECTURE.md`, `docs/CONTEXT_MANAGEMENT.md`, `docs/MEMORY.md`, `docs/TOOLS.md`, `docs/TESTING.md`, `docs/PERFORMANCE.md`, `docs/SECURITY.md`, `docs/DEPENDENCIES.md`.
- **Compact Context Packs (`docs/context/*.context.md`)**: Ultra-compact (<200 lines) context files for rapid AI agent onboarding.
- **Architecture Decision Records (`docs/decisions/`)**: Recorded ADR-0001 through ADR-0008.
- **Experiment Tracking Registry (`experiments/`)**: Structured directories and markdown template for model training runs.

### Changed
- **Enriched Troubleshooting (`docs/TROUBLESHOOTING.md`)**: Structured diagnostics for CUDA OOM, non-finite gradients, tokenizer fingerprints, and generation loops.
- **Expanded Configuration Reference (`docs/CONFIGURATION.md`)**: Comprehensive parameter catalog covering all YAML fields and environment variables.

### Removed
- **Generated Build Metadata (`src/llm_engine.egg-info`)**: Removed generated setuptools directory from working tree; gitignored via `*.egg-info/`.
- **Stale Scratch Artifacts**: Cleaned stale temporary write files (`reports/.finetuning.json.a_l4qm6d`) and empty placeholder directories (`.agents`, `.codex`).

---

## [0.1.0] — Initial Platform Release
- Initial configuration-driven GPT model, tokenizer, training, evaluation, generation, export, and serving implementation.
