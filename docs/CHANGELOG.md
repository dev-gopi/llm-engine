## 2026-09-21 — Full task-registry and repository audit

- Audited the unpacked repository against all 97 task-registry sections and reconciled stale local statuses against source/tests.
- Fixed duplicate task IDs (`EMB-001` → `EMB-002` for the Embeddings API and the later `OPS-001` → `OPS-003` for health/readiness) and removed undefined dependency references.
- Added an automated task-registry integrity audit with regression coverage.
- Fixed pytest test-module collisions, MTP trainer wiring, reasoning/QLoRA/long-context profile issues, training-report CPU sampling, missing deterministic reasoning/long-context fixtures, and dataset-catalog coverage.
- Converted machine-specific documentation links to portable repository-relative links and replaced the stale phase roadmap with an authoritative-registry-derived view.
- Verified 833 tests pass with 1 intentional source-archive skip; five additional data-preparation modules require the declared but unavailable `pyarrow` dependency in this offline sandbox.
- Registry after reconciliation: 97 unique IDs, 81 `COMPLETED`, 16 evidence-gated `TODO`. See `docs/AUDIT_2026-09-21.md`.

## 2026-09-21 — RAG/API contracts and protected lifecycle

- Completed RAG-005 with a candidate-pool/reranker pipeline, optional neural SentenceTransformers cross-encoder adapter, metadata/citation preservation, and measurable recall/MRR/NDCG evaluation.
- Hardened API-004 usage reporting so the native `/v1/generate` response also reports cached and reasoning tokens.
- Hardened API-005 so unsupported presence/frequency penalties are rejected instead of silently ignored.
- Completed API-007 with runtime-derived capability discovery and architecture/context metadata at `/v1/models` and `/v1/models/{model_id}/capabilities`.
- Completed API-008 with dedicated-admin-key protected `/admin/models/load`, `/admin/models/unload`, and `/admin/models/reload`, atomic backend lifecycle transitions, concurrent-drain behavior, and audit events.


## 2026-09-21 — API/runtime contracts

- Added strict JSON Schema structured outputs with schema compatibility and final validation/refusal states.
- Added OpenAI-compatible `/v1/responses` generation and SSE event contracts while retaining Chat Completions.
- Added configurable reasoning-effort runtime budgets and reasoning-token accounting.
- Added first-class local/HTTP MCP execution contracts with shared authorization and approval gates.
- Added RAG reranker interface and optional candidate reranking.
- Added standardized token/cache usage accounting and OpenAI generation-parameter validation.
- Added client-disconnect and explicit request cancellation with cleanup propagation.
- Added regression tests for structured outputs, Responses API, reasoning budgets, MCP, reranking, parameters, and cancellation.

## 2026-09-21 — DATA-004..DATA-007 data evaluation infrastructure

- Added independent train/evaluation exact and near-duplicate contamination audit tooling (DATA-004).
- Added deterministic per-document quality scoring and optional quality-weighted mixture support (DATA-005).
- Added recorded-mixture ablation comparison against capability metrics without ranking or causal claims (DATA-006).
- Added packed-sequence and dynamic-padding efficiency accounting (DATA-007).
- Added focused regression coverage for all four data-track additions.
- DATA-004 remains TODO until the actual governed train/evaluation corpora are available for an independent run; no empty or synthetic audit is treated as production evidence.

# Changelog (`docs/CHANGELOG.md`)

Notable user-visible changes and architectural milestones are recorded here.

---

## [Unreleased] — 2026-09-19: AI Agent Optimization, Audit & Engineering Knowledge System

### Added
- **CTX-002 Long-Context Validation Infrastructure**: Added paired 2K/4K/8K model and pretraining profiles, deterministic governed long-document retrieval fixtures, checkpoint-backed long-context evaluation mode, and peak-memory measurement tooling. Long-context capability remains evidence-gated until matching checkpoints pass retrieval and memory validation.
- **RSN-002 Reasoning SFT Data and Training Profile**: Added governed repository-authored math, code, logic, planning, verification, and self-correction SFT seed corpora; a deterministic reasoning training profile; strict `<thinking>...</thinking>` boundary validation; and an assistant-only trainer policy that requires explicit loss masks.
- **TRAIN-004 Training Stability Diagnostics**: Trainer histories and resume
  state now record loss, gradient and parameter scales, a gradient-to-parameter
  update-scale proxy, and bounded logit activation diagnostics without retaining
  tensor snapshots.
- **TRAIN-003 Hyperparameter Sweep Framework**: Added deterministic,
  config-driven Cartesian plans for LR, batch, scheduler, optimizer, and
  related training axes. The sweep CLI is dry-run by default and requires an
  explicit flag to execute sequential trials.
- **TRAIN-001 Scaling-Law Framework**: Added deterministic validation and
  log-compute trend summaries for versioned experiment observations.
- **TRAIN-002 Compute and Token Accounting**: Added deterministic trainer
  reports for supervised tokens, FLOP estimates, device-hours, and throughput.
- **DOC-001 Capability-State Reconciliation**: Corrected stale documentation
  claims about active context lengths, bounded sequential tool orchestration,
  session memory, and regression gates; unsupported autonomous/parallel and
  trained-quality claims remain explicitly future or externally validated.
- **EVAL-002 Release Capability Matrix**: Added a versioned mandatory gate
  covering knowledge, math, code, reasoning, instruction following, structured
  JSON, tools, RAG, long context, safety, hallucination, and refusal behavior.
- **AGT-003 Tool-Use Reliability Suite**: Added governed, versioned fixtures
  for selection, argument schemas, result/error handling, parallel intent, and
  denied calls, while retaining bounded sequential runtime execution.
- **CHAT-002 Instruction Quality Contract**: Added canonical untrusted
  tool-message turns (excluded from supervised loss), corpus-review guidance,
  and a versioned deterministic five-category instruction-following suite.
- **SCALE-002 Progressive Model Growth**: Completed tested checkpoint growth
  from 16 to 32 layers with exact initialization-time output preservation;
  continued-pretraining convergence remains an external operational check.
- **CUR-001 Resume Compatibility**: Restoring an ungrouped sampler now
  correctly ignores its serialized empty curriculum-weight list, including
  when the batch size changes between checkpoints.
- **CUR-001 Curriculum and Diagnostics**: Added validated epoch-indexed
  source-mixture schedules with resume-safe sampler weights, plus training
  history/checkpoint metrics for gradient clipping and output magnitudes.
- **DATA-002 Versioned Streaming Mixtures**: Added validated, license-aware
  source declarations with retained quality metrics and deterministic weighted
  streaming provenance. The web/code/math/reasoning profile remains inactive
  until its reviewed local artifacts are supplied.
- **ATT-001 Hardware-Aware Attention Selection**: Added explicit `auto`,
  `sdpa`, and `eager` attention policies. Compatible CUDA tensors delegate to
  PyTorch SDPA/FlashAttention dispatch; CPU and unsupported inputs use the
  deterministic eager fallback.
- **REG-001 Content-Addressed Export Manifests**: Export bundles now include a
  deterministic manifest with model/config/tokenizer hashes, tokenizer
  fingerprint, compatibility configuration, format, and precision metadata.
- **MEM-001 Privacy-Bounded Session Memory**: Added deterministic, session-only
  memory retrieval and explicitly enabled, API-key-protected endpoints to read
  or delete persisted session memory. The interface is disabled by default.
- **API-001 Session Context API**: Added opt-in, API-key-protected context
  capacity metrics and explicit session-history compaction. Tool definitions,
  uploaded files, and tool results remain request-scoped and are never
  persisted for telemetry.
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

## 2026-09-21 — Evaluation, embeddings, observability and release gates

- Added OpenAI-compatible API conformance, tool lifecycle conformance, and structured-output conformance suites.
- Added dedicated `/v1/embeddings` service with deterministic development encoder, batching, dimensions, usage, and metadata discovery.
- Added prefix-cache hit/miss, cached-token, prefill-saved, memory, and eviction metrics.
- Added `/health`, `/ready`, and enriched metrics coverage.
- Added RAG recall/reranking/faithfulness/citation benchmark primitives and release benchmark script.
- Added checkpoint evaluation/regression/promotion pipeline with release-candidate report generation.
- Added model card, dataset card, and reproducibility manifest generation.
- Added capability evidence gating so client-visible capability flags can be tied to explicit validation evidence.
