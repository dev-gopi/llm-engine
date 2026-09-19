# Directory Map & File Responsibilities (`docs/DIRECTORY_MAP.md`)

This map documents the purpose and role of every primary directory and file in the `llm-engine` repository.

---

## 1. Directory Tree Overview

```text
llm-engine/
├── AGENTS.md                # Primary AI agent entry point & working protocol
├── pyproject.toml           # Packaging definition, dependencies, pytest configuration
├── README.md                # Human-facing project overview
├── Dockerfile               # Production container image definition
├── compose.yaml             # Docker Compose service specification
├── compose.gpu.yaml         # GPU-enabled Compose service specification
│
├── configs/                 # Declarative YAML runtime and training configurations
│   ├── defaults/            # Reusable baseline configuration fragments
│   ├── diffusion/           # Experimental diffusion & image generation configs
│   ├── scaling/             # Future large-scale cluster & MoE configs
│   ├── text/                # Experimental future text model architectures (1B, 7B, 30B)
│   ├── vision/              # Vision-language multimodal configurations
│   ├── model.gpu.yaml       # Primary active GPU model architecture configuration
│   ├── pretraining.gpu.yaml # Pretraining hyperparameters, datasets, optimizer
│   ├── finetuning.gpu.yaml  # Supervised fine-tuning (SFT) configuration
│   └── inference.yaml       # Serving, generation, and KV cache runtime config
│
├── src/                     # Core Python library source code
│   ├── model/               # Decoder-only Transformer, attention, RoPE, RMSNorm, SwiGLU
│   ├── tokenizer/           # Custom Byte-Pair Encoding (BPE) engine
│   ├── datasets/            # Sharded streaming dataloaders, collators, governance
│   ├── optim/               # AdamW optimizer, schedulers, EMA helper
│   ├── training/            # Trainer engine, checkpointing, distributed sync, reports
│   ├── post_training/       # Direct Preference Optimization (DPO) and preference data
│   ├── inference/           # Generator, Paged KV cache, sampling, quantization
│   ├── serving/             # FastAPI REST endpoints, WebSocket streaming, batch queue
│   ├── mcp/                 # Model Context Protocol client & tool execution
│   ├── vision/              # Vision encoders and patch embeddings
│   ├── multimodal/          # Multimodal projection bridge
│   ├── image_data/          # Image processing and dataset loaders
│   ├── diffusion/           # Experimental diffusion pipeline & UNet
│   └── utils/               # Device detection, logging, config parsing, seeding
│
├── scripts/                 # Operational CLI commands & utility scripts
│   ├── train.py             # Main entry point for pretraining and fine-tuning
│   ├── train_dpo.py         # Entry point for Direct Preference Optimization
│   ├── serve.py             # Start production FastAPI & WebSocket server
│   ├── chat.py              # Interactive terminal chat interface
│   ├── tokenize.py          # Train BPE tokenizer and process raw text
│   ├── build_token_shards.py# Pack tokenized datasets into binary memory-mapped shards
│   ├── evaluate.py          # Run validation loss and perplexity evaluation
│   ├── export.py            # Export checkpoints to safetensors, ONNX, GGUF
│   └── ...                  # Dataset preparation and audit scripts
│
├── tests/                   # 670+ automated pytest suites
│   ├── fixtures/            # Test mock servers and synthetic datasets
│   ├── test_gpt.py          # Core model forward pass tests
│   ├── test_attention.py    # GQA and causal attention tests
│   ├── test_training_system.py # End-to-end training and checkpoint resume tests
│   └── ...                  # Comprehensive unit tests for all modules
│
├── docs/                    # Persistent engineering knowledge base & ADRs
│   ├── context/             # Compact context packs for AI agents
│   ├── decisions/           # Architecture Decision Records (ADRs)
│   └── *.md                 # Subsystem technical reference documents
│
├── data/                    # Datasets and cache directory (mostly gitignored)
│   ├── raw/                 # Raw downloaded source texts (gitignored)
│   ├── processed/           # Tokenized and formatted JSONL datasets (gitignored)
│   ├── shards/              # Memory-mapped binary token shards (gitignored)
│   ├── tokenizer/           # Base 40K vocabulary and merge files (gitignored)
│   ├── tokenizer-finetuning/# Extended 42K vocabulary files (gitignored)
│   └── rag/                 # SQLite document embeddings and index (gitignored)
│
├── checkpoints/             # Training weight checkpoints (gitignored)
│   ├── pretraining/         # Pretraining latest.pt and best.pt
│   └── finetuning/          # Fine-tuning latest.pt and best.pt
│
├── reports/                 # Live training reports and evaluation metrics
│   ├── training_report.html # Interactive HTML dashboard with live charts
│   └── training_report.json # JSON metrics stream for monitoring
│
├── experiments/             # Structured experiment tracking registry
│   ├── active/              # Currently running experiment manifests
│   ├── completed/           # Completed experiment logs and evaluations
│   └── templates/           # Experiment manifest templates
│
└── logs/                    # Training and server log files (gitignored)
```

---

## 2. Source Code Invariants (`src/`)

- Every module within `src/` must be importable as a top-level namespace package or via `llm_engine` packages.
- Zero cyclic dependencies between `src/model/` and `src/training/`. `src/model/` must never import from `src/training/` or `src/serving/`.
- All model hyperparameters must be parameterized; no hardcoded dimension constants in computational graph modules.

