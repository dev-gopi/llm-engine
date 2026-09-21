# Changelog

Notable user-visible changes and architectural milestones are recorded here.

---

## [Unreleased] — 2026-09-19: AI Agent Optimization, Audit & Engineering Knowledge System

### Added
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

## 0.1.0

- Initial configuration-driven GPT model, tokenizer, training, evaluation, generation, export, and serving implementation.

## 2026-09-21 — Modern model/runtime enhancement audit

- Added opt-in hybrid linear/dense attention with fixed-state causal linear decoding; default model/checkpoint configuration remains unchanged.
- Added sparse-MoE router load-balancing objective and bounded routing diagnostics, disabled by default.
- Added Q1_0-style 1.125-bit engine-native research packing/export with explicit non-GGUF compatibility labeling.
- Added analytical deployment memory planning across weight/KV precisions, context and batch size, with API/CLI access.
- Added OpenAI-compatible external inference backend and GGUF llama.cpp launcher so specialized low-bit models can run behind the Gopi API without duplicating custom kernels.
- Expanded runtime capability metadata and upgraded the playground with reasoning-effort/model/resource visibility.
- Expanded the live training report with model architecture, deployment footprints, MTP auxiliary loss and MoE auxiliary loss.
- Added modern 8K hybrid and planning-only 7B hybrid-MoE profiles plus research documentation grounded in current published model cards.
