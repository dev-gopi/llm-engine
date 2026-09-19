# LLM Engine Knowledge Base (`docs/README.md`)

Welcome to the engineering documentation system and persistent project memory for `llm-engine`. This documentation is structured for **progressive context loading** by AI coding agents and human engineers.

---

## 🚀 AI Agent Quick Start

If you are an AI coding agent:
1. **Start at [`AGENTS.md`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/AGENTS.md)** at the root of the repository.
2. Look up specific subsystems in [`docs/PROJECT_INDEX.md`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/PROJECT_INDEX.md).
3. Read the relevant compact context pack in [`docs/context/`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/context/) before opening source files.
4. Do **not** scan the entire repository.

---

## 📚 Documentation Navigation Map

### High-Level Context & Architecture
- **[PROJECT_CONTEXT.md](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/PROJECT_CONTEXT.md)**: Engineering mission, design principles, hardware limits (RTX 3050 4GB).
- **[ARCHITECTURE.md](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/ARCHITECTURE.md)**: High-level architectural diagrams, component interactions, and layers.
- **[CURRENT_STATE.md](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/CURRENT_STATE.md)**: Exact verified implementation parameters, shapes, and metrics.
- **[TARGET_STATE.md](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/TARGET_STATE.md)**: Target production capabilities, roadmap milestones, and migration notes.
- **[DIRECTORY_MAP.md](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/DIRECTORY_MAP.md)**: Complete file tree and directory responsibility definitions.
- **[COMPONENTS.md](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/COMPONENTS.md)**: Detailed breakdown of every Python package and class hierarchy.
- **[DATA_FLOW.md](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/DATA_FLOW.md)**: End-to-end data pipelines from raw corpus to token logits and serving streams.

### Subsystem Specifications
- **[MODEL.md](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/MODEL.md)**: Decoder mathematics, GQA, RoPE, RMSNorm, SwiGLU, and weight tying.
- **[TOKENIZER.md](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/TOKENIZER.md)**: BPE algorithm, byte fallbacks, special tokens, vocabulary expansion.
- **[DATASETS.md](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/DATASETS.md)**: Shard storage, streaming dataloader, dataset mixture weights, governance.
- **[TRAINING.md](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/TRAINING.md)**: Training loops, AdamW, cosine scheduling, EMA, mixed precision, chunked loss.
- **[EVALUATION.md](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/EVALUATION.md)**: Perplexity tracking, domain validation, retention gating, regression tests.
- **[INFERENCE.md](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/INFERENCE.md)**: KV caching, Paged KV blocks, sampling algorithms, INT8 quantization.
- **[SERVING.md](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/SERVING.md)**: FastAPI endpoints, WebSocket streaming protocol, dynamic batching.
- **[AGENT_ARCHITECTURE.md](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/AGENT_ARCHITECTURE.md)**: Autonomous agent loops, planning, RAG, and memory.
- **[CONTEXT_MANAGEMENT.md](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/CONTEXT_MANAGEMENT.md)**: Context budgets, prompt compaction, and prefix caching.
- **[MEMORY.md](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/MEMORY.md)**: Ephemeral and persistent conversation state in SQLite.
- **[TOOLS.md](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/TOOLS.md)**: Local tool definitions, MCP client integration, safety sandbox.
- **[SCALING.md](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/SCALING.md)**: Model scaling tiers (80M → 1T), FSDP, tensor parallelism, LoRA, model growth.
- **[VISION.md](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/VISION.md)**: Vision encoder, multimodal VLM, diffusion pipeline, image data processing.

### Operational Guides & Governance
- **[CONFIGURATION.md](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/CONFIGURATION.md)**: Complete reference of all parameters in `configs/*.yaml`.
- **[DEPENDENCIES.md](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/DEPENDENCIES.md)**: Python dependencies, system libraries, and CUDA requirements.
- **[TESTING.md](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/TESTING.md)**: Pytest suite guide, regression benchmarks, running tests.
- **[DEPLOYMENT.md](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/DEPLOYMENT.md)**: Docker containerization, reverse proxy, TLS, production startup.
- **[PERFORMANCE.md](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/PERFORMANCE.md)**: Memory profiling, throughput optimization, 4GB VRAM tuning.
- **[SECURITY.md](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/SECURITY.md)**: Prompt injection protection, tool sandboxing, authentication.
- **[TROUBLESHOOTING.md](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/TROUBLESHOOTING.md)**: Known error signatures, CUDA OOM remedies, NaN loss fixes.
- **[UPGRADE_ROADMAP.md](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/UPGRADE_ROADMAP.md)**: 10-phase upgrade roadmap.
- **[TASKS.md](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/TASKS.md)**: Task backlog and issue tracker.
- **[CHANGELOG.md](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/CHANGELOG.md)**: Release and modification history.
- **[decisions/](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/decisions/)**: Architecture Decision Records (ADRs).
- **[context/](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/context/)**: Ultra-compact context packs for AI agents.

