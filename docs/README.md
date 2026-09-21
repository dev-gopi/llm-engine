# LLM Engine Knowledge Base (`docs/README.md`)

Welcome to the engineering documentation system and persistent project memory for `llm-engine`. This documentation is structured for **progressive context loading** by AI coding agents and human engineers.

---

## 🚀 AI Agent Quick Start

If you are an AI coding agent:
1. **Start at [`AGENTS.md`](../AGENTS.md)** at the root of the repository.
2. Look up specific subsystems in [`docs/PROJECT_INDEX.md`](PROJECT_INDEX.md).
3. Read the relevant compact context pack in [`docs/context/`](context) before opening source files.
4. Do **not** scan the entire repository.

---

## 📚 Documentation Navigation Map

### High-Level Context & Architecture
- **[PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)**: Engineering mission, design principles, hardware limits (RTX 3050 4GB).
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: High-level architectural diagrams, component interactions, and layers.
- **[CURRENT_STATE.md](CURRENT_STATE.md)**: Exact verified implementation parameters, shapes, and metrics.
- **[TARGET_STATE.md](TARGET_STATE.md)**: Target production capabilities, roadmap milestones, and migration notes.
- **[DIRECTORY_MAP.md](DIRECTORY_MAP.md)**: Complete file tree and directory responsibility definitions.
- **[COMPONENTS.md](COMPONENTS.md)**: Detailed breakdown of every Python package and class hierarchy.
- **[DATA_FLOW.md](DATA_FLOW.md)**: End-to-end data pipelines from raw corpus to token logits and serving streams.

### Subsystem Specifications
- **[MODEL.md](MODEL.md)**: Decoder mathematics, GQA, RoPE, RMSNorm, SwiGLU, and weight tying.
- **[TOKENIZER.md](TOKENIZER.md)**: BPE algorithm, byte fallbacks, special tokens, vocabulary expansion.
- **[DATASETS.md](DATASETS.md)**: Shard storage, streaming dataloader, dataset mixture weights, governance.
- **[TRAINING.md](TRAINING.md)**: Training loops, AdamW, cosine scheduling, EMA, mixed precision, chunked loss.
- **[EVALUATION.md](EVALUATION.md)**: Perplexity tracking, domain validation, retention gating, regression tests.
- **[INFERENCE.md](INFERENCE.md)**: KV caching, Paged KV blocks, sampling algorithms, INT8 quantization.
- **[SERVING.md](SERVING.md)**: FastAPI endpoints, WebSocket streaming protocol, dynamic batching.
- **[AGENT_ARCHITECTURE.md](AGENT_ARCHITECTURE.md)**: Autonomous agent loops, planning, RAG, and memory.
- **[CONTEXT_MANAGEMENT.md](CONTEXT_MANAGEMENT.md)**: Context budgets, prompt compaction, and prefix caching.
- **[MEMORY.md](MEMORY.md)**: Ephemeral and persistent conversation state in SQLite.
- **[TOOLS.md](TOOLS.md)**: Local tool definitions, MCP client integration, safety sandbox.
- **[SCALING.md](SCALING.md)**: Model scaling tiers (80M → 1T), FSDP, tensor parallelism, LoRA, model growth.
- **[VISION.md](VISION.md)**: Vision encoder, multimodal VLM, diffusion pipeline, image data processing.

### Operational Guides & Governance
- **[CONFIGURATION.md](CONFIGURATION.md)**: Complete reference of all parameters in `configs/*.yaml`.
- **[DEPENDENCIES.md](DEPENDENCIES.md)**: Python dependencies, system libraries, and CUDA requirements.
- **[TESTING.md](TESTING.md)**: Pytest suite guide, regression benchmarks, running tests.
- **[DEPLOYMENT.md](DEPLOYMENT.md)**: Docker containerization, reverse proxy, TLS, production startup.
- **[PERFORMANCE.md](PERFORMANCE.md)**: Memory profiling, throughput optimization, 4GB VRAM tuning.
- **[SECURITY.md](SECURITY.md)**: Prompt injection protection, tool sandboxing, authentication.
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)**: Known error signatures, CUDA OOM remedies, NaN loss fixes.
- **[UPGRADE_ROADMAP.md](UPGRADE_ROADMAP.md)**: Derived roadmap of the remaining evidence-gated work.
- **[TASKS.md](TASKS.md)**: Authoritative task registry, status, dependencies, acceptance criteria, and validation commands.
- **[AUDIT_2026-09-21.md](AUDIT_2026-09-21.md)**: Full repository/task-registry audit and validation result.
- **[CHANGELOG.md](CHANGELOG.md)**: Release and modification history.
- **[decisions/](decisions)**: Architecture Decision Records (ADRs).
- **[context/](context)**: Ultra-compact context packs for AI agents.

