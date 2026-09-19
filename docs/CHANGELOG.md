# Changelog (`docs/CHANGELOG.md`)

Notable user-visible changes and architectural milestones are recorded here.

---

## [Unreleased] — 2026-09-19: AI Agent Optimization, Audit & Engineering Knowledge System

### Added
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
