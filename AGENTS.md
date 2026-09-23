# AGENTS.md — AI Coding Agent Entry Point & Working Protocol

> **CRITICAL DIRECTIVE FOR AI AGENTS:**
> **DO NOT SCAN OR READ THE ENTIRE REPOSITORY.**
> This project is structured with a progressive context documentation system. Reading the entire repository wastes context tokens and risks hallucinations.
> Follow this entry point, identify your task, look up the relevant component in [`docs/PROJECT_INDEX.md`](docs/PROJECT_INDEX.md), then read **only** the linked source, configuration, test, and guide files needed for that task.

---

## 1. Project Purpose & Mission

`llm-engine` is a modular, high-efficiency Large Language Model (LLM) framework engineered in Python and PyTorch. It provides a complete lifecycle for foundation decoder transformers:
- **Data Engineering**: Ingestion, tokenization, streaming shard preparation, filtering, and governance.
- **Model Architecture**: Compact causal decoder-only Transformer with Grouped-Query Attention (GQA), Rotary Position Embeddings (RoPE), SwiGLU feed-forward networks, and RMSNorm.
- **Training**: Pretraining, supervised fine-tuning (SFT), Direct Preference Optimization (DPO), chunked cross-entropy loss, and mixed precision.
- **Inference & Serving**: Autoregressive generation with KV caching (including Paged KV cache), WebSocket streaming, OpenAI-compatible FastAPI endpoints, and local/MCP tool-use orchestration.

The long-term goal is to evolve this compact foundation model into an agentic, reasoning-capable, long-context platform with hybrid attention and tool-calling execution, optimized for consumer GPU hardware (e.g. NVIDIA RTX 3050 4GB).

---

## 2. Hardware Boundary & Operating Constraints

- **Primary Development Hardware**: NVIDIA GeForce RTX 3050 (4 GB VRAM), 16–32 GB Host RAM.
- **Architectural Philosophy**: **Efficiency > Architecture > Data Quality > Training Stability > Inference Optimization > Parameter Count**.
- **Memory Invariants**:
  1. Always enable `gradient_checkpointing: true` in active GPU training configs.
  2. Use small per-device batch sizes (e.g., `batch_size: 2`) combined with gradient accumulation (e.g., `gradient_accumulation_steps: 16` or `32`).
  3. Use mixed precision (`fp16` or `bf16`).
  4. Use Grouped-Query Attention (`heads: 8`, `kv_heads: 2`) to minimize KV cache footprint.
  5. Never load full raw datasets into memory; use `lazy_dataset: true` and memory-mapped token shards (`src/datasets/token_shards.py`).

---

## 3. Quick Architecture Summary

```
                       ┌────────────────────────────┐
                       │       Client / UI          │
                       └─────────────┬──────────────┘
                                     │ HTTP / WebSocket
                                     ▼
                       ┌────────────────────────────┐
                       │   FastAPI & Serving Layer  │ (src/serving/)
                       └─────────────┬──────────────┘
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           ▼                         ▼                         ▼
┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐
│ Inference Runtime  │    │  MCP / Tool System │    │ RAG & Embeddings   │
│ (src/inference/)   │    │ (src/mcp/, tools)  │    │ (src/inference/rag)│
└──────────┬─────────┘    └────────────────────┘    └────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────────────────────────┐
│                          Core Model (src/model/)                       │
│  - GPTDecoder: 16 Layers, hidden_size=512, heads=8, kv_heads=2 (GQA)   │
│  - Rotary Positional Embedding (RoPE), RMSNorm, SwiGLU                 │
│  - Tied Word Embeddings (40,000 base -> 42,000 extended)               │
└────────────────────────────────────────────────────────────────────────┘
           ▲
           │ Checkpoints & Weights
┌──────────┴─────────┐
│ Training Pipeline  │ (src/training/, src/post_training/, src/optim/)
│ - Trainer & Sched  │
│ - Loss & Metrics   │
└────────────────────┘
```

- **Active Model Size**: 81,314,304 parameters (~81.3M) at 40K vocab; 82,338,304 (~82.3M) at 42K vocab.
- **Context Length**: Configured maximum 1024 tokens; pretraining/fine-tuning sequences currently 512 tokens.

---

## 4. Directory Map

| Path | Purpose | Authoritative Status |
| :--- | :--- | :--- |
| `src/model/` | Transformer blocks, attention, RoPE, RMSNorm, SwiGLU, model config | Source of Truth for model architecture |
| `src/tokenizer/` | Byte-Pair Encoding (BPE) encoder, decoder, trainer | Source of Truth for tokenization |
| `src/datasets/` | Shard reader, streaming loaders, collators, governance filters | Source of Truth for data loading |
| `src/training/` | Trainer loop, checkpointing, distributed sync, evaluator | Source of Truth for training orchestration |
| `src/optim/` | AdamW optimizer, EMA helper, learning rate schedulers | Source of Truth for optimization |
| `src/post_training/` | Direct Preference Optimization (DPO), preference datasets | Source of Truth for alignment |
| `src/inference/` | Generator, KV cache, Paged KV cache, quantization, safety | Source of Truth for inference runtime |
| `src/serving/` | FastAPI REST API, WebSocket streaming server, batching | Source of Truth for serving |
| `src/mcp/` | Model Context Protocol (MCP) client, tool execution | Source of Truth for tool integration |
| `src/vision/` | Vision Transformer (ViT) encoder, classifier, patch embedding | Source of Truth for vision models |
| `src/multimodal/` | Vision-Language wrapper (MiniGPT + ViT), vision projector | Source of Truth for multimodal integration |
| `src/diffusion/` | Pixel-space & latent diffusion pipelines, U-Net, VAE, scheduler | Source of Truth for image generation |
| `src/image_data/` | Pillow image processing, folder datasets, dataset auditing | Source of Truth for image data |
| `configs/` | YAML configuration files for models, training, serving, vision, diffusion | Source of Truth for runtime parameters |
| `scripts/` | Executable CLI scripts for training, tokenizing, serving, eval | CLI operational entry points |
| `tests/` | 670+ automated pytest suites | Regression & contract verification |
| `docs/` | AI-agent knowledge base, specifications, ADRs, context packs | Project memory |
| `experiments/` | Experiment tracking manifests, logs, and notes | Experiment records |

---

## 5. Standard Commands

Always use the project's virtual environment (`.venv/bin/...`):

### Running Tests
```bash
# Run all fast tests
.venv/bin/pytest -q

# Run specific test modules
.venv/bin/pytest tests/test_model_config.py tests/test_attention.py -q
.venv/bin/pytest tests/test_training_system.py -q
```

### Pretraining & Fine-Tuning
```bash
# Pretraining
.venv/bin/python scripts/train.py --config configs/pretraining.gpu.yaml

# Fine-Tuning / SFT
.venv/bin/python scripts/train.py --config configs/finetuning.gpu.yaml

# DPO Training
.venv/bin/python scripts/train_dpo.py --config configs/dpo.gpu.yaml
```

### Tokenization & Data Preparation
```bash
# Train BPE Tokenizer
.venv/bin/python scripts/tokenize.py --train --data data/raw/corpus.txt --output data/tokenizer/

# Build binary token shards
.venv/bin/python scripts/build_token_shards.py --input data/processed/ --output data/shards/
```

### Inference & Serving
```bash
# Start API & WebSocket server
.venv/bin/python scripts/serve.py --config configs/inference.yaml

# Terminal interactive chat
.venv/bin/python scripts/chat.py --checkpoint checkpoints/finetuning/best.pt
```

---

## 6. Rules for Modifying the Project

1. **Check Compatibility First**: Never modify model hyperparameters (hidden dimensions, layer count, head count, vocab size) without verifying compatibility with existing checkpoints and running [`tests/test_vocabulary_compatibility.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/tests/test_vocabulary_compatibility.py).
2. **Never Break Working Tests**: All 670+ existing tests must continue to pass. Run targeted tests before editing and full tests before committing.
3. **No Blind Source Code Edits**: Always identify the authoritative source file (see [`docs/PROJECT_INDEX.md`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/PROJECT_INDEX.md)) before making modifications.
4. **Configuration Integrity**: Parameter values belong in `configs/*.yaml`. Do not hardcode magic numbers in source files.
5. **Update Project Memory**:
   - If an architectural decision is made, record an ADR in [`docs/decisions/`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/decisions/).
   - If a roadmap task is started or completed, update [`docs/TASKS.md`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/TASKS.md).
   - If user-visible behavior changes, document it in [`docs/CHANGELOG.md`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/CHANGELOG.md).

---

## 7. AI Agent Task Execution Workflow

When tasked with any development job, follow this exact 8-step protocol:

```text
1. Read AGENTS.md (this file)
           │
           ▼
2. Find Task / Subsystem in docs/PROJECT_INDEX.md
           │
           ▼
3. Read the linked guide and relevant source/configuration/test files
           │
           ▼
4. Inspect Authoritative Source Files (e.g. src/<subsystem>/...)
           │
           ▼
5. Implement Smallest Safe Change
           │
           ▼
6. Run Specific Tests (.venv/bin/pytest tests/test_<subsystem>.py)
           │
           ▼
7. Update Documentation / TASKS.md / CHANGELOG.md
           │
           ▼
8. Report Concise Diff & Validation Output
```

---

## 8. Documentation Quick-Navigation

- **Lookup Matrix**: [`docs/PROJECT_INDEX.md`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/PROJECT_INDEX.md)
- **Current Architecture Snapshot**: [`docs/CURRENT_STATE.md`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/CURRENT_STATE.md)
- **Target Architecture & Plan**: [`docs/TARGET_STATE.md`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/TARGET_STATE.md)
- **Phased Upgrade Roadmap**: [`docs/UPGRADE_ROADMAP.md`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/UPGRADE_ROADMAP.md)
- **Active Task Backlog**: [`docs/TASKS.md`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/TASKS.md)
- **Configuration Reference**: [`docs/CONFIGURATION.md`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/CONFIGURATION.md)
- **Troubleshooting Guide**: [`docs/TROUBLESHOOTING.md`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/TROUBLESHOOTING.md)
- **Architecture Decisions**: [`docs/decisions/README.md`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/decisions/README.md)

