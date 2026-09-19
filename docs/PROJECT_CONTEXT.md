# Project Context & Engineering Philosophy (`docs/PROJECT_CONTEXT.md`)

## 1. Mission Statement

The `llm-engine` project is a ground-up implementation of a production-grade, modular, decoder-only Large Language Model platform. The objective is to build a high-performance system that can progress from foundation pretraining to instruction tuning, reasoning, autonomous tool execution, and local deployment on constrained consumer hardware.

Instead of relying on monolithic, opaque third-party libraries, `llm-engine` exposes every layer of the LLM stack cleanly:
- Exact PyTorch tensor operations for attention, normalization, and activations.
- Transparent memory management (Paged KV caching, gradient checkpointing, chunked cross-entropy).
- Native streaming dataloaders for multi-domain token shards.
- Production serving protocols (WebSocket token streaming, FastAPI REST endpoints).
- Direct MCP and local tool-calling execution loops.

---

## 2. Core Engineering Principles

### Principle 1: Efficiency Over Raw Scale
On consumer hardware (e.g. 4 GB VRAM), scaling model parameter count blindly produces Out-Of-Memory (OOM) failures or throttled throughput. Architectural efficiency—such as Grouped-Query Attention (GQA), Rotary Position Embeddings (RoPE), SwiGLU FFNs, chunked loss computation, and Paged KV caches—takes precedence over parameter size.

### Principle 2: Strict Progressive Context for AI Agents
AI coding agents must not consume thousands of tokens scanning the entire repository. Every subsystem is self-documenting and referenced by [`docs/PROJECT_INDEX.md`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/PROJECT_INDEX.md) and compact context packs in [`docs/context/`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/context/).

### Principle 3: Single Source of Truth
No values are duplicated across documentation and source code:
- Model dimensions and architecture: [`src/model/config.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/model/config.py) & [`configs/model.gpu.yaml`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/configs/model.gpu.yaml).
- Training defaults: [`configs/pretraining.gpu.yaml`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/configs/pretraining.gpu.yaml).
- Runtime behavior: authoritative Python modules in `src/`.

### Principle 4: Contract Integrity & Regression Gating
The repository maintains over 670 automated unit and integration tests. Changes to public interfaces, tokenizers, checkpoints, or loss calculation must maintain backward compatibility or pass explicit migration verification (`tests/test_vocabulary_compatibility.py`, `tests/test_evaluation_regressions.py`).

---

## 3. Hardware Boundary: The 4 GB VRAM Standard

All training and inference routines must run stably on an **NVIDIA GeForce RTX 3050 (4 GB VRAM)**.

### The 4 GB VRAM Budget:
```text
┌────────────────────────────────────────────────────────┐
│ Total VRAM Ceiling: 4,096 MiB                          │
├────────────────────────────────────────────────────────┤
│ [165 MB] Model Weights (FP16 / BF16, 81.3M params)     │
│ [990 MB] AdamW Master State (FP32 weights, m, v)       │
│ [165 MB] EMA Shadow Weights (FP16 / BF16)              │
│ [850 MB] Activation Memory (Gradient Checkpointing ON)│
│ [600 MB] PyTorch / CUDA Overhead & Scratch Workspace   │
├────────────────────────────────────────────────────────┤
│ Free Margin: ~1,300 MiB (Headroom for dynamic batches) │
└────────────────────────────────────────────────────────┘
```

If gradient checkpointing is turned OFF, activation memory for 16 layers with sequence length 512 exceeds 3,500 MiB, immediately triggering an Out-of-Memory (OOM) error. Gradient checkpointing is therefore an absolute non-negotiable invariant for GPU training.

