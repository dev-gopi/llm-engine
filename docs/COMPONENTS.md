# Component Catalog & Dependency Relationships (`docs/COMPONENTS.md`)

This document analyzes the primary Python components in `src/`, their operational purpose, dependency graph, and maintenance status.

---

## 1. Model Subsystem (`src/model/`)

### `GPTModel` (`src/model/gpt.py`)
- **What it does**: Top-level causal decoder-only Transformer module. Manages token embeddings, an arbitrary stack of `TransformerBlock` layers, final RMSNorm, and the language modeling head (`lm_head`).
- **Why it exists**: Provides the unified PyTorch model contract (`forward(input_ids, attention_mask, position_ids, ...) -> ModelOutput(logits, loss, ...)`).
- **Dependencies**: `TransformerBlock`, `EmbeddingLayer`, `RMSNorm`, `RotaryEmbedding`.
- **Dependents**: `Trainer`, `Generator`, `ChatSession`, `ModelExporter`.
- **Status**: **Actively used**. Foundation of the entire platform.

### `TransformerBlock` (`src/model/transformer_block.py`)
- **What it does**: Implements a single pre-norm or post-norm Transformer decoder layer with residual connections.
- **Why it exists**: Encapsulates self-attention and feed-forward sublayers with configurable dropout, residual scaling, and gradient checkpointing.
- **Dependencies**: `CausalSelfAttention`, `FeedForward`, `RMSNorm`.
- **Dependents**: `GPTModel`.
- **Status**: **Actively used**. Scheduled for hybrid attention extensions in Phase 6.

### `CausalSelfAttention` (`src/model/attention.py`)
- **What it does**: Implements Multi-Head Attention (MHA) and Grouped-Query Attention (GQA). Computes scaled dot-product attention with causal triangular masking and rotary positional encoding.
- **Why it exists**: Primary autoregressive sequence modeling mechanism. GQA drastically reduces KV cache memory for serving.
- **Dependencies**: `RotaryEmbedding`.
- **Dependents**: `TransformerBlock`.
- **Status**: **Actively used**. Target for FlashAttention integration in Phase 4.

### `FeedForward` (`src/model/feed_forward.py`)
- **What it does**: Implements the position-wise feed-forward network. Supports SwiGLU, GeGLU, GELU, ReLU, and MoE routing foundations.
- **Why it exists**: Non-linear channel mixing and representation capacity expansion.
- **Dependencies**: PyTorch `nn.Module`.
- **Dependents**: `TransformerBlock`.
- **Status**: **Actively used**. MoE sparse routing will be added in Phase 9.

### `RotaryEmbedding` (`src/model/positional.py`)
- **What it does**: Computes rotary position embeddings (RoPE) and applies complex tensor rotations to Query and Key projections.
- **Why it exists**: Allows relative position modeling without absolute position embedding parameter lookup tables.
- **Dependencies**: PyTorch tensor operations.
- **Dependents**: `CausalSelfAttention`.
- **Status**: **Actively used**. Scheduled for NTK-aware and YaRN scaling in Phase 6.

### `ChunkedCrossEntropyLoss` (`src/model/loss.py`)
- **What it does**: Calculates cross-entropy loss by chunking logits along the sequence dimension, optionally adding z-loss penalty for logit stability.
- **Why it exists**: Prevents allocating the massive `[batch, seq_len, vocab_size]` logit tensor in FP32 all at once, saving hundreds of megabytes of peak VRAM during backward pass.
- **Dependencies**: PyTorch functional operations.
- **Dependents**: `GPTModel`, `Trainer`.
- **Status**: **Actively used**. Critical for 4 GB VRAM training stability.

---

## 2. Tokenization Subsystem (`src/tokenizer/`)

### `BPETokenizer` (`src/tokenizer/bpe.py`)
- **What it does**: Byte-level Pair Encoding tokenizer with regex pre-tokenization, special token handling, and UTF-8 byte fallback.
- **Why it exists**: Converts raw strings to model token IDs and vice versa without out-of-vocabulary (OOV) failures.
- **Dependencies**: Python standard library `regex`.
- **Dependents**: `Trainer`, `Dataloader`, `Generator`, `ChatSession`.
- **Status**: **Actively used**. Extended to 42K tokens in fine-tuning.

### `TokenizerTrainer` (`src/tokenizer/trainer.py`)
- **What it does**: Trains a new BPE vocabulary and merge table from raw text files.
- **Why it exists**: Allows training domain-specific tokenizers from scratch.
- **Dependencies**: `BPETokenizer`.
- **Dependents**: `scripts/tokenize.py`.
- **Status**: **Actively used**.

---

## 3. Data Subsystem (`src/datasets/`)

### `ShardedDatasetLoader` (`src/datasets/token_shards.py`, `src/datasets/loader.py`)
- **What it does**: Streams binary token shards using memory mapping (`numpy.memmap`). Supports dynamic dataset mixture weighting and sequence packing.
- **Why it exists**: Avoids loading huge datasets into host RAM, providing infinite non-blocking batch sampling.
- **Dependencies**: `numpy`, PyTorch `IterableDataset`.
- **Dependents**: `Trainer`.
- **Status**: **Actively used**.

### `GovernanceFilters` (`src/datasets/filters.py`, `src/datasets/governance.py`)
- **What it does**: Performs document-level quality scoring, PII redaction, toxicity filtering, and deduplication.
- **Why it exists**: Enforces data safety and license compliance prior to tokenization.
- **Dependencies**: Python standard library.
- **Dependents**: `scripts/clean_jsonl_corpus.py`, `scripts/audit_datasets.py`.
- **Status**: **Actively used**. Scheduled for MinHash LSH expansion in Phase 1.

---

## 4. Training Subsystem (`src/training/`, `src/optim/`)

### `Trainer` (`src/training/trainer.py`)
- **What it does**: Orchestrates the training loop, gradient accumulation, AMP GradScaler, EMA shadow updates, evaluation triggers, and checkpoint saves.
- **Why it exists**: Unified execution engine for pretraining and fine-tuning.
- **Dependencies**: `GPTModel`, `AdamW`, `LRScheduler`, `Evaluator`, `CheckpointManager`.
- **Dependents**: `scripts/train.py`.
- **Status**: **Actively used**.

### `Evaluator` (`src/training/evaluator.py`)
- **What it does**: Evaluates the model across held-out domain datasets, calculating loss, cross-entropy, and perplexity.
- **Why it exists**: Objective measurement of generalization and retention loss during training.
- **Dependencies**: `GPTModel`, `DataLoader`.
- **Dependents**: `Trainer`, `scripts/evaluate.py`.
- **Status**: **Actively used**.

### `DPOTrainer` (`src/post_training/dpo.py`)
- **What it does**: Implements Direct Preference Optimization loss comparing chosen vs. rejected response completions relative to a frozen reference model.
- **Why it exists**: Post-training preference alignment without training a separate reward model.
- **Dependencies**: `GPTModel`.
- **Dependents**: `scripts/train_dpo.py`.
- **Status**: **Actively used**. Target for full production alignment in Phase 3.

---

## 5. Inference & Serving Subsystem (`src/inference/`, `src/serving/`)

### `AutoregressiveGenerator` (`src/inference/generator.py`)
- **What it does**: Performs step-by-step autoregressive generation with KV caching, temperature, top-p, min-p, and repetition penalties.
- **Why it exists**: Core text generation engine.
- **Dependencies**: `GPTModel`, `PagedKVCache`, `Sampler`.
- **Dependents**: `FastAPIServer`, `WebSocketServer`, `ChatSession`.
- **Status**: **Actively used**. Speculative decoding will be added in Phase 7.

### `PagedKVCache` (`src/inference/paged_kv_cache.py`)
- **What it does**: Allocates KV cache memory in fixed-size blocks (pages), supporting dynamic sequence growth and prefix sharing.
- **Why it exists**: Eliminates internal and external memory fragmentation during multi-user serving.
- **Dependencies**: PyTorch tensor operations.
- **Dependents**: `AutoregressiveGenerator`.
- **Status**: **Actively used**. Prefix caching expansion in Phase 4.

### `FastAPIServer` & `WebSocketServer` (`src/serving/api.py`, `src/serving/websocket.py`)
- **What it does**: Exposes asynchronous HTTP endpoints and WebSocket streams for real-time generation.
- **Why it exists**: Production serving interface for web clients and external applications.
- **Dependencies**: `fastapi`, `uvicorn`, `pydantic`.
- **Dependents**: `scripts/serve.py`, `ui/`.
- **Status**: **Actively used**.

---

## 6. Scaling & Distributed Training (`src/training/`)

### `ModelGrowth` (`src/training/model_growth.py`)
- **What it does**: Progressive model scaling via depth doubling and vocabulary expansion. Initializes new layers with identity residual mappings.
- **Why it exists**: Enables incremental model growth from small validated checkpoints to larger architectures without training from scratch.
- **Dependencies**: `GPTModel`, PyTorch checkpoint utilities.
- **Dependents**: `scripts/grow_checkpoint.py`.
- **Status**: **Actively used**. Foundation for the scaling roadmap from 80M to 1B+ parameters.

### `LoRALinear` & `apply_lora` (`src/training/peft.py`)
- **What it does**: Implements Low-Rank Adaptation (LoRA) by injecting trainable rank-decomposed weight matrices into frozen linear layers.
- **Why it exists**: Enables parameter-efficient fine-tuning with <1% trainable parameters, critical for 4 GB VRAM adaptation.
- **Dependencies**: PyTorch `nn.Module`.
- **Dependents**: `Trainer`, fine-tuning scripts.
- **Status**: **Actively used**. Core PEFT method for memory-constrained fine-tuning.

### `DistributedTopology` (`src/training/multinode.py`)
- **What it does**: Parses torchrun environment variables to construct the multi-node process topology (world_size, rank, local_rank, node_rank).
- **Why it exists**: Provides clean topology abstraction for FSDP and DDP distributed training.
- **Dependencies**: Python standard library, PyTorch distributed.
- **Dependents**: `Trainer`, `DistributedCheckpoint`.
- **Status**: **Actively used**. Required for multi-GPU and multi-node training.

### `ElasticTraining` (`src/training/elastic.py`)
- **What it does**: Implements elastic fault tolerance for distributed training, allowing workers to join and leave dynamically.
- **Why it exists**: Production training resilience for long-running jobs on unreliable infrastructure.
- **Dependencies**: `DistributedTopology`, PyTorch elastic.
- **Dependents**: `Trainer`.
- **Status**: **Infrastructure ready**. Planned for production scaling.

### `DistributedCheckpoint` (`src/training/distributed_checkpoint.py`)
- **What it does**: Saves and loads model state across multiple GPU ranks using sharded checkpoint files.
- **Why it exists**: Full model state cannot fit in single-rank memory at large scales; sharded saving avoids OOM.
- **Dependencies**: `DistributedTopology`, PyTorch distributed.
- **Dependents**: `Trainer`.
- **Status**: **Actively used**. Essential for FSDP training.

### `TensorParallel` (`src/inference/tensor_parallel.py`)
- **What it does**: Splits linear layers across multiple GPUs for intra-layer tensor parallelism during inference.
- **Why it exists**: Enables serving models too large for a single GPU's memory.
- **Dependencies**: PyTorch distributed.
- **Dependents**: `AutoregressiveGenerator`.
- **Status**: **Actively used**. Key for inference-time scaling.

---

## 7. Vision & Multimodal Subsystem (`src/vision/`, `src/multimodal/`)

### `VisionEncoder` (`src/vision/encoder.py`)
- **What it does**: Vision Transformer (ViT) that encodes fixed-size RGB images as patch token sequences with a pooled CLS token.
- **Why it exists**: Provides visual understanding backbone for multimodal vision-language integration.
- **Dependencies**: `PatchEmbedding`, PyTorch `nn.Module`.
- **Dependents**: `VisionClassifier`, `VisionLanguageModel`.
- **Status**: **Actively used**. Foundation for Phase 8 vision integration.

### `VisionClassifier` (`src/vision/classifier.py`)
- **What it does**: Image classification head that applies a linear projection over pooled VisionEncoder features.
- **Why it exists**: Enables supervised vision encoder pretraining and standalone classification tasks.
- **Dependencies**: `VisionEncoder`.
- **Dependents**: Vision training scripts.
- **Status**: **Actively used**.

### `VisionProjector` (`src/multimodal/projector.py`)
- **What it does**: Two-layer MLP (Linear → GELU → Dropout → Linear → LayerNorm) mapping visual features to LLM embedding dimensions.
- **Why it exists**: Bridges the dimension gap between vision encoder hidden size and language model hidden size.
- **Dependencies**: PyTorch `nn.Module`.
- **Dependents**: `VisionLanguageModel`.
- **Status**: **Actively used**.

### `VisionLanguageModel` (`src/multimodal/model.py`)
- **What it does**: Non-invasive wrapper connecting VisionEncoder to MiniGPT for vision-language tasks. Concatenates visual and text embeddings with loss masking.
- **Why it exists**: Adds multimodal capability without modifying MiniGPT source or breaking existing text checkpoints.
- **Dependencies**: `VisionEncoder`, `VisionProjector`, `MiniGPT`.
- **Dependents**: Multimodal training scripts.
- **Status**: **Actively used**. Core of Phase 8 multimodal roadmap.

---

## 8. Diffusion & Image Generation (`src/diffusion/`, `src/image_data/`)

### `SmallUNet` (`src/diffusion/unet.py`)
- **What it does**: Compact conditional U-Net noise predictor with 2 encoder/decoder levels, spatial self-attention, and optional cross-attention.
- **Why it exists**: Predicts Gaussian noise for DDPM/DDIM denoising while staying within 4 GB VRAM.
- **Dependencies**: PyTorch `nn.Module`.
- **Dependents**: `DiffusionPipeline`, `LatentDiffusionPipeline`.
- **Status**: **Actively used**.

### `DiffusionScheduler` (`src/diffusion/scheduler.py`)
- **What it does**: Implements DDPM forward and reverse diffusion processes with linear or cosine beta schedules, plus DDIM fast sampling.
- **Why it exists**: Manages the noise schedule for training and inference.
- **Dependencies**: PyTorch tensor operations.
- **Dependents**: `DiffusionPipeline`, `LatentDiffusionPipeline`.
- **Status**: **Actively used**.

### `DiffusionPipeline` (`src/diffusion/pipeline.py`)
- **What it does**: Orchestrates diffusion training loss computation and iterative sampling with classifier-free guidance.
- **Why it exists**: Unified pipeline for pixel-space diffusion training and inference.
- **Dependencies**: `SmallUNet`, `DiffusionScheduler`.
- **Dependents**: `LatentDiffusionPipeline`, diffusion training scripts.
- **Status**: **Actively used**.

### `AutoencoderKL` (`src/diffusion/vae.py`)
- **What it does**: Convolutional variational autoencoder that compresses images to a lower-dimensional latent space.
- **Why it exists**: Enables latent diffusion which is far more efficient than pixel-space diffusion.
- **Dependencies**: PyTorch `nn.Module`.
- **Dependents**: `LatentDiffusionPipeline`.
- **Status**: **Actively used**.

### `DiffusionTextEncoder` (`src/diffusion/text_encoder.py`)
- **What it does**: Small Transformer encoder that maps token IDs to conditioning vectors for text-guided diffusion.
- **Why it exists**: Provides text conditioning without requiring external CLIP models.
- **Dependencies**: PyTorch `nn.Module`.
- **Dependents**: `LatentDiffusionPipeline`.
- **Status**: **Actively used**.

### `LatentDiffusionPipeline` (`src/diffusion/latent_pipeline.py`)
- **What it does**: Combines VAE encoding, latent-space diffusion, and text-conditioned generation in a single pipeline.
- **Why it exists**: Full text-to-image generation pipeline optimized for consumer GPU.
- **Dependencies**: `AutoencoderKL`, `SmallUNet`, `DiffusionScheduler`, `DiffusionTextEncoder`.
- **Dependents**: Diffusion training/inference scripts.
- **Status**: **Actively used**.

### `ImageProcessor` & `ImageDataset` (`src/image_data/`)
- **What it does**: Pillow-based image loading, resizing, cropping, augmentation, and normalization. Folder-based datasets for training.
- **Why it exists**: Dependency-light image pipeline that avoids requiring torchvision.
- **Dependencies**: `Pillow`, `numpy`.
- **Dependents**: `VisionEncoder` training, `DiffusionPipeline` training.
- **Status**: **Actively used**.
