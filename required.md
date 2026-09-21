# LLM Engine

A modular, scalable LLM engineering platform for building, training, fine-tuning, evaluating, optimizing, and serving modern Large Language Models.

The project starts with a compact decoder-only Transformer designed for local GPU development and progressively evolves into a complete LLM platform supporting:

* Foundation-model pretraining
* High-quality data processing
* Custom tokenization
* Instruction tuning
* Reasoning
* Preference alignment
* Long-context processing
* Efficient attention
* KV caching
* Quantization
* Speculative decoding
* Structured generation
* Function/tool calling
* RAG
* Embeddings
* Agents
* Vision and multimodal models
* Distributed training
* Production inference
* OpenAI-compatible APIs
* Evaluation
* Observability
* Safety
* Model versioning
* Reproducible experiments

---

# Table of Contents

1. [Project Vision](#1-project-vision)
2. [Project Status](#2-project-status)
3. [Current Model](#3-current-model)
4. [Feature Classification](#4-feature-classification)
5. [Complete LLM Lifecycle](#5-complete-llm-lifecycle)
6. [Model Architecture](#6-model-architecture)
7. [Tokenizer](#7-tokenizer)
8. [Dataset Engineering](#8-dataset-engineering)
9. [Pretraining](#9-pretraining)
10. [Training Infrastructure](#10-training-infrastructure)
11. [Instruction Tuning](#11-instruction-tuning)
12. [Reasoning](#12-reasoning)
13. [Preference Alignment](#13-preference-alignment)
14. [Long Context](#14-long-context)
15. [Attention Architecture](#15-attention-architecture)
16. [KV Cache](#16-kv-cache)
17. [Inference & Decoding](#17-inference--decoding)
18. [Speculative Decoding](#18-speculative-decoding)
19. [Quantization](#19-quantization)
20. [Structured Generation](#20-structured-generation)
21. [Tool Calling](#21-tool-calling)
22. [RAG](#22-rag)
23. [Embeddings & Reranking](#23-embeddings--reranking)
24. [Agent System](#24-agent-system)
25. [Memory](#25-memory)
26. [Vision & Multimodal](#26-vision--multimodal)
27. [Fine-Tuning](#27-fine-tuning)
28. [Distributed Training](#28-distributed-training)
29. [Evaluation](#29-evaluation)
30. [Safety](#30-safety)
31. [Model Export](#31-model-export)
32. [Serving](#32-serving)
33. [Production Infrastructure](#33-production-infrastructure)
34. [Observability](#34-observability)
35. [Reproducibility](#35-reproducibility)
36. [Fault Tolerance](#36-fault-tolerance)
37. [Model Registry](#37-model-registry)
38. [Security](#38-security)
39. [Testing](#39-testing)
40. [Project Architecture](#40-project-architecture)
41. [Development Roadmap](#41-development-roadmap)
42. [Priority Matrix](#42-priority-matrix)
43. [Hardware Strategy](#43-hardware-strategy)
44. [Target Architecture](#44-target-architecture)
45. [Final Goal](#45-final-goal)

---

# 1. Project Vision

The project is not intended to be only a Transformer implementation.

The goal is to build a complete LLM engineering stack:

```text
                    ┌──────────────────────────┐
                    │       Data Layer         │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │    Tokenization Layer    │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │     Training Engine       │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │      Foundation LLM      │
                    └────────────┬─────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
        Instruction          Reasoning           Domain
          Tuning              Training          Adaptation
              │                  │                  │
              └──────────────────┼──────────────────┘
                                 ▼
                         ┌───────────────┐
                         │   Chat LLM    │
                         └───────┬───────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
             RAG                Tools             Vision
              │                  │                  │
              └──────────────────┼──────────────────┘
                                 ▼
                         ┌───────────────┐
                         │ Agent Runtime │
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │Inference Engine│
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │ Production API│
                         └───────────────┘
```

---

# 2. Project Status

The current project already contains the foundation of a custom LLM engine.

### Current status

* [x] Decoder-only Transformer
* [x] Causal language modeling
* [x] RoPE
* [x] GQA
* [x] RMSNorm
* [x] SwiGLU
* [x] Weight tying
* [x] Custom tokenizer
* [x] Pretraining
* [x] WikiText dataset
* [x] TinyStories dataset
* [x] Dataset weighting
* [x] Validation
* [x] Cross entropy
* [x] Perplexity
* [x] Checkpointing
* [x] Gradient checkpointing
* [x] GPU training
* [x] DDP foundation
* [x] Training reports
* [x] Text generation
* [x] WebSocket serving foundation
* [x] Dataset cleaning, MinHash deduplication, PII/secret redaction, and governance audits
* [x] Chat SFT, recovery SFT, DPO, and fixed-prompt/domain evaluation
* [x] KV caching, prefix caching, paged KV accounting, dynamic batching, and CPU INT8 inference
* [x] OpenAI-compatible API, SSE/WebSocket streaming, authentication, rate limiting, metrics, and health checks
* [x] RAG, web search, MCP integration, sparse MoE, model growth, and vision-projector profile
* [~] Bounded sequential tool orchestration and session-scoped lexical memory;
  neither establishes autonomous planning/reflection, parallel tools, or
  trained model reliability.

Status markers in the following sections are evidence-based: `[x]` implemented,
`[~]` partial/optional or requiring workload-specific validation, and `[ ]` future.

---

# 3. Current Model

Current architecture:

```yaml
architecture: decoder-only Transformer

hidden_size: 512
layers: 16

attention_heads: 8
kv_heads: 2

ffn_hidden_size: 2048
ffn_activation: swiglu

normalization: rms_norm

position_encoding: rotary
rope_base: 10000
rope_scale: 1.0

max_position: 512

vocab_size: 40000
runtime_vocab_size: ~42000

tie_word_embeddings: true

gradient_checkpointing: true
embedding_dropout: 0.0
initializer_range: 0.02
```

Current training configuration includes:

```yaml
batch_size: 2
epochs: 5
seed: 42

max_sequence_length: 1024

num_workers: 4
pin_memory: true
persistent_workers: true
prefetch_factor: 2

lazy_dataset: true

distributed_strategy: ddp
checkpoint_format: single_file
```

Current dataset mixture:

```text
WikiText      90%
TinyStories   10%
```

---

# 4. Feature Classification

Not every possible LLM feature is mandatory.

The project divides capabilities into four categories.

## Tier 1 — Foundation

These are essential.

```text
Model architecture
Tokenizer
Dataset pipeline
Pretraining
Validation
Checkpointing
Evaluation
Inference
KV cache
Training monitoring
Reproducibility
```

## Tier 2 — Production LLM

These should be implemented after the foundation is stable.

```text
Instruction tuning
Chat templates
Structured generation
Quantization
Tool calling
RAG
Embeddings
Fine-tuning
Production API
Continuous batching
Observability
Security
```

## Tier 3 — Advanced

These improve efficiency and capabilities.

```text
Long context
RoPE scaling
FlashAttention
Paged KV cache
Speculative decoding
MTP
Hybrid attention
Linear attention
Agent runtime
DPO
Advanced distributed training
```

## Tier 4 — Research

These are optional research directions.

```text
MoE
1-bit/ternary models
2-bit quantization
Custom CUDA kernels
Vision architecture
Audio architecture
Video architecture
Extreme long context
Expert parallelism
```

---

# 5. Complete LLM Lifecycle

A complete model lifecycle:

```text
                    RAW DATA
                       │
                       ▼
              DATA COLLECTION
                       │
                       ▼
              DATA CLEANING
                       │
                       ▼
             DEDUPLICATION
                       │
                       ▼
             QUALITY FILTERING
                       │
                       ▼
             TOKENIZATION
                       │
                       ▼
                 PRETRAINING
                       │
                       ▼
                BASE MODEL
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
         SFT        Continued      Domain
                    Training       Training
          │            │            │
          └────────────┼────────────┘
                       ▼
                  CHAT MODEL
                       │
          ┌────────────┼─────────────┐
          ▼            ▼             ▼
      Reasoning       DPO           Tools
          │            │             │
          └────────────┼─────────────┘
                       ▼
                  ALIGNED LLM
                       │
          ┌────────────┼─────────────┐
          ▼            ▼             ▼
         RAG         Agents        Vision
          │            │             │
          └────────────┼─────────────┘
                       ▼
              INFERENCE ENGINE
                       │
          ┌────────────┼─────────────┐
          ▼            ▼             ▼
       Quantized      Cached       Batched
       Inference    Inference    Inference
                       │
                       ▼
                 PRODUCTION API
```

---

# 6. Model Architecture

## Current Architecture

```text
Input Tokens
     │
     ▼
Token Embedding
     │
     ▼
┌──────────────────────────────┐
│ Transformer Block × 16       │
│                              │
│ RMSNorm                      │
│    ↓                         │
│ GQA + RoPE                   │
│    ↓                         │
│ Residual                     │
│    ↓                         │
│ RMSNorm                      │
│    ↓                         │
│ SwiGLU                       │
│    ↓                         │
│ Residual                     │
└──────────────────────────────┘
     │
     ▼
LM Head
     │
     ▼
Logits
```

## Future Architecture

```text
Input
 │
 ▼
Embedding
 │
 ▼
Hybrid Transformer
 │
 ├── Full Attention
 ├── Linear Attention
 ├── GQA
 ├── RoPE
 ├── RMSNorm
 ├── SwiGLU
 ├── Residual Connections
 └── Optional MoE
 │
 ▼
MTP Heads
 │
 ▼
LM Head
```

---

# 7. Tokenizer

The tokenizer converts text into token IDs.

```text
Text
 │
 ▼
Tokenizer
 │
 ▼
Token IDs
 │
 ▼
Transformer
```

### Current

* [x] Custom tokenizer
* [x] Vocabulary
* [x] Token IDs

### Planned

* [x] Byte-level BPE evaluation
* [ ] Multilingual optimization
* [x] Special chat tokens
* [ ] Tool tokens
* [x] Thinking tokens
* [ ] Vision tokens
* [x] Tokenizer fingerprinting and append-only compatibility checks

Potential special tokens:

```text
<|system|>
<|user|>
<|assistant|>
<|tool|>
<|thinking|>
<|end|>
```

---

# 8. Dataset Engineering

The quality of training data is one of the most important factors in model quality.

## Pipeline

```text
Raw Data
   │
   ▼
Collection
   │
   ▼
Parsing
   │
   ▼
Language Detection
   │
   ▼
Quality Filtering
   │
   ▼
Deduplication
   │
   ▼
PII Filtering
   │
   ▼
Safety Filtering
   │
   ▼
Tokenization
   │
   ▼
Packing
   │
   ▼
Dataset Shards
   │
   ▼
Training
```

### Dataset categories

* [x] WikiText
* [x] TinyStories
* [x] Optional educational web text
* [ ] Books
* [ ] Wikipedia
* [ ] Scientific text
* [ ] Technical documentation
* [x] Optional code pretraining data
* [ ] Mathematics
* [ ] Multilingual text
* [x] Conversation and instruction data
* [ ] Reasoning data
* [ ] Tool-use data
* [ ] Vision-language data

### Data quality

* [x] Exact deduplication
* [x] Near-duplicate detection (MinHash LSH)
* [x] Language detection and low-quality filtering
* [x] PII and secret redaction
* [ ] Toxicity filtering
* [x] Validation-contamination detection and audit reports
* [~] Dataset manifests and audit fingerprints

---

# 9. Pretraining

The foundation training objective is next-token prediction.

```text
"The cat is"
       │
       ▼
Predict next token
       │
       ▼
"sleeping"
```

Mathematically:

```text
P(x_t | x_1, x_2, ..., x_(t-1))
```

### Required training components

* [x] Causal LM
* [x] Cross entropy
* [x] Perplexity
* [x] Batch training
* [x] Validation
* [x] Checkpoints

### Planned improvements

* [ ] Larger datasets
* [ ] Better dataset mixture
* [ ] Dynamic dataset weighting
* [ ] Curriculum learning
* [ ] Longer sequences
* [x] Cosine scheduling with warmup
* [x] Gradient clipping
* [x] BF16/FP16 mixed precision
* [x] NaN/Inf detection
* [x] Gradient norm monitoring

---

# 10. Training Infrastructure

The training engine should support:

```text
Dataset
   │
   ▼
DataLoader
   │
   ▼
Batch
   │
   ▼
Forward Pass
   │
   ▼
Loss
   │
   ▼
Backward Pass
   │
   ▼
Gradient Accumulation
   │
   ▼
Optimizer
   │
   ▼
Scheduler
   │
   ▼
Checkpoint
```

### Features

* [x] GPU training
* [x] Gradient accumulation
* [x] Gradient checkpointing
* [x] DDP foundation
* [x] Checkpointing
* [x] Resume
* [x] BF16
* [x] FP16 / AMP
* [x] Gradient clipping
* [x] Gradient norm monitoring
* [ ] Activation monitoring
* [x] NaN/Inf detection
* [x] Optimizer and scheduler state recovery

---

# 11. Instruction Tuning

Pretraining creates a base language model.

Instruction tuning turns it into an instruction-following model.

```text
Base Model
    │
    ▼
Instruction Dataset
    │
    ▼
Supervised Fine-Tuning
    │
    ▼
Chat Model
```

Example:

```text
User:
Explain how HTTP works.

Assistant:
HTTP is an application-layer protocol...
```

### Features

* [x] Instruction datasets
* [x] Chat template with system, user, assistant, and end delimiters
* [x] Prompt loss masking and multi-turn conversations
* [x] SFT and recovery SFT
* [x] Instruction/fixed-prompt evaluation

---

# 12. Reasoning

Reasoning requires dedicated training.

```text
Base Model
    │
    ▼
Reasoning Dataset
    │
    ▼
Reasoning SFT
    │
    ▼
Reasoning Model
```

Domains:

* [ ] Mathematics
* [ ] Logic
* [ ] Programming
* [ ] Multi-step reasoning
* [ ] Planning
* [ ] Verification
* [ ] Self-correction

Possible format:

```text
<thinking>
Internal reasoning representation
</thinking>

<answer>
Final answer
</answer>
```

A special `<thinking>` token by itself does not create reasoning ability. The behavior must be learned from suitable training data and objectives.

---

# 13. Preference Alignment

After SFT:

```text
Base Model
    ↓
SFT
    ↓
Preference Alignment
    ↓
Aligned Model
```

Potential approaches:

* [x] DPO
* [ ] ORPO
* [ ] IPO
* [ ] Reward model
* [ ] RLHF
* [ ] RLAIF

A practical initial target is DPO.

---

# 14. Long Context

Current context is approximately:

```text
Training sequence: 512
Architecture max position: 1024
```

The architecture/configuration should eventually be made consistent.

Roadmap:

```text
1K
 │
 ▼
2K
 │
 ▼
4K
 │
 ▼
8K
 │
 ▼
16K
 │
 ▼
32K
 │
 ▼
64K+
```

Required:

* [ ] Larger max position
* [ ] Longer training sequences
* [x] Opt-in RoPE scaling policies (linear, NTK-aware, YaRN); extended-context training is unverified
* [ ] Long-context data
* [ ] Long-context evaluation
* [ ] Efficient attention
* [ ] Efficient KV cache

---

# 15. Attention Architecture

## Current

Standard full attention.

## Future options

### Full Attention

Best general-purpose baseline.

### Sliding Window Attention

Restricts attention to a local window.

### Linear Attention

Reduces attention complexity for long sequences.

### Sparse Attention

Only selected tokens interact.

### Hybrid Attention

Combines multiple attention mechanisms.

Example:

```text
Layer 1 → Linear
Layer 2 → Linear
Layer 3 → Full
Layer 4 → Linear
Layer 5 → Linear
Layer 6 → Full
...
```

### Advanced

* [ ] FlashAttention
* [ ] Linear attention
* [ ] Sliding window
* [ ] Sparse attention
* [ ] Chunked attention
* [ ] Hybrid attention

---

# 16. KV Cache

Autoregressive inference repeatedly processes previous context.

KV caching avoids recomputing previous keys and values.

```text
Prompt
  │
  ▼
K/V
  │
  ▼
KV Cache
  │
  ▼
Next Token
```

Planned:

* [x] Basic GPU KV cache
* [x] Block-paged KV cache accounting
* [x] Prefix caching and KV reuse
* [ ] INT8 KV cache
* [ ] INT4 KV cache
* [ ] Cache eviction
* [ ] Cache statistics

---

# 17. Inference & Decoding

Generation should support:

* [x] Greedy decoding, temperature, Top-K, Top-P, and Min-P
* [ ] Repetition penalty
* [ ] Frequency penalty
* [ ] Presence penalty
* [x] Stop sequences, EOS handling, seeded sampling, streaming, and batched generation

Example:

```text
Prompt
  │
  ▼
Model
  │
  ▼
Logits
  │
  ▼
Sampler
  │
  ▼
Token
  │
  └──────────────┐
                 ▼
              Model
```

---

# 18. Speculative Decoding

Instead of using the large model for every token:

```text
Large Model
   ↓
1 token
   ↓
Large Model
   ↓
1 token
```

use:

```text
             Draft Model
                  │
                  ▼
             Several Tokens
                  │
                  ▼
             Main Model
                  │
                  ▼
          Verify / Accept
```

Features:

* [ ] Draft model
* [ ] Candidate generation
* [ ] Target verification
* [ ] Acceptance algorithm
* [ ] MTP
* [ ] Benchmarking
* [ ] Adaptive speculation

---

# 19. Quantization

Quantization reduces memory and inference cost.

Supported target levels:

```text
FP32
 │
 ▼
BF16 / FP16
 │
 ▼
INT8
 │
 ▼
INT4
 │
 ▼
2-bit
 │
 ▼
1-bit / ternary
```

Planned:

* [ ] FP16 export
* [ ] BF16 export
* [x] CPU dynamic INT8
* [ ] INT4
* [x] Dynamic weight quantization
* [ ] Activation quantization
* [ ] KV quantization
* [ ] GGUF
* [ ] GPTQ
* [ ] AWQ

### Research

* [ ] 2-bit
* [ ] 1-bit
* [ ] Ternary weights
* [ ] Group-wise scaling
* [ ] Custom low-bit kernels

---

# 20. Structured Generation

The model should be able to produce guaranteed structured outputs.

Example:

```json
{
  "name": "John",
  "age": 30
}
```

Features:

* [~] JSON tool-call envelopes (validated after generation)
* [x] JSON schema validation, enum constraints, and typed tool arguments
* [ ] Token-level grammar/logit constraints

This is especially important for tool calling and agents.

---

# 21. Tool Calling

Architecture:

```text
User
 │
 ▼
LLM
 │
 ▼
Tool Selection
 │
 ▼
Structured Arguments
 │
 ▼
Tool Executor
 │
 ▼
Tool Result
 │
 ▼
LLM
 │
 ▼
Final Answer
```

Example:

```json
{
  "tool": "get_weather",
  "arguments": {
    "city": "Kolkata"
  }
}
```

Features:

* [x] Function and tool schemas
* [~] Model-directed selection and argument generation (strictly validated after generation)
* [x] Local and MCP tool execution, result handling, and structured errors
* [ ] Multiple tools
* [x] Bounded sequential tool calls with allowlists and structured errors
* [ ] Parallel tool calls
* [~] Versioned tool-use contract/evaluation fixtures; trained tool-use quality is unverified

---

# 22. RAG

Retrieval-Augmented Generation:

```text
User Query
    │
    ▼
Embedding
    │
    ▼
Vector Search
    │
    ▼
Retrieved Documents
    │
    ▼
Reranker
    │
    ▼
Context
    │
    ▼
LLM
    │
    ▼
Answer
```

Components:

* [x] Document ingestion, parsing, chunking, and retrieval
* [~] Local lexical retrieval and optional PDF/document RAG
* [ ] Reranking
* [x] Context assembly and source-aware prompt construction
* [~] Retrieval-result source references
* [ ] Hybrid search

---

# 23. Embeddings & Reranking

## Embeddings

```text
Text
 ↓
Embedding Model
 ↓
Vector
```

Applications:

* Semantic search
* RAG
* Similarity
* Classification
* Clustering

## Reranking

```text
Query
 +
Candidates
 ↓
Reranker
 ↓
Ranked Documents
```

Features:

* [ ] Dedicated embedding model
* [x] Local similarity/lexical search for RAG
* [ ] Reranker
* [ ] Hybrid retrieval

---

# 24. Agent System

The agent layer sits above the LLM.

```text
User
 │
 ▼
Planner
 │
 ▼
LLM
 │
 ▼
Tool
 │
 ▼
Observation
 │
 ▼
LLM
 │
 ▼
Next Action
 │
 ▼
Result
```

Features:

* [ ] Planning
* [ ] Task decomposition
* [ ] Tool selection
* [ ] Tool execution
* [ ] Observation handling
* [ ] Agent state
* [ ] Error recovery
* [ ] Multi-step workflows
* [ ] RAG
* [ ] Memory
* [ ] Human approval checkpoints

---

# 25. Memory

Memory should be implemented primarily at the application/runtime layer.

Types:

```text
Short-term conversation memory
Long-term user memory
Semantic memory
Episodic memory
Task memory
```

Features:

* [x] Persistent conversation history with expiration and explicit export approval
* [ ] Summarization
* [ ] Persistent memory
* [ ] Vector memory
* [x] Session-scoped lexical memory retrieval and expiration
* [x] Explicit opt-in API access/deletion and training-export approval

---

# 26. Vision & Multimodal

Current / partial:

```text
Text → LLM → Text
Image → Vision encoder → projector → LLM → Text (small adapter profile)
```

Future:

```text
Image
  │
  ▼
Vision Encoder
  │
  ▼
Projector
  │
  ▼
LLM
  │
  ▼
Text
```

Features:

* [x] Vision encoder, image projector, and visual tokens
* [ ] OCR
* [ ] Image understanding
* [ ] Visual Q&A
* [ ] Image-text datasets
* [ ] Multimodal SFT

Future extensions:

* [ ] Audio
* [ ] Speech
* [ ] Video
* [ ] Multimodal agents

---

# 27. Fine-Tuning

Fine-tuning options:

```text
Full Fine-Tuning
      │
      ├── LoRA
      ├── QLoRA
      └── Adapters
```

Features:

* [x] Full fine-tuning
* [x] LoRA adapter training
* [ ] QLoRA
* [x] Adapter training
* [~] Adapter merging/export integration
* [x] Domain and instruction fine-tuning

For limited GPU hardware, LoRA/QLoRA should be prioritized.

---

# 28. Distributed Training

Current:

```text
DDP
```

Future:

```text
DDP
 │
 ▼
FSDP
 │
 ▼
ZeRO
 │
 ▼
Tensor Parallelism
 │
 ▼
Pipeline Parallelism
 │
 ▼
Sequence Parallelism
 │
 ▼
Expert Parallelism
```

These become important only as model size and hardware scale increase.

---

# 29. Evaluation

Loss and perplexity alone are insufficient.

## Language

* [x] Perplexity and held-out validation

## Knowledge

* [x] Fixed-prompt benchmark runner

## Reasoning

* [ ] Mathematics
* [ ] Logic
* [ ] Multi-step reasoning

## Code

* [ ] Code completion
* [ ] Code generation

## Instruction

* [x] Instruction-following and multi-turn regression cases

## Long Context

* [ ] Needle-in-haystack
* [ ] Long-document understanding
* [ ] Context retrieval

## Tool Use

* [ ] Function calling
* [ ] Tool selection
* [ ] Argument correctness

## RAG

* [ ] Retrieval recall
* [ ] Answer correctness
* [ ] Citation correctness

## Safety

* [ ] Safety evaluations
* [ ] Jailbreak testing
* [ ] Prompt-injection testing

Every model version should have an evaluation report.

---

# 30. Safety

Safety exists at multiple layers.

```text
Dataset
   │
   ▼
Training
   │
   ▼
Model
   │
   ▼
Inference
   │
   ▼
Application
```

Features:

* [ ] Dataset filtering
* [ ] Safety SFT
* [ ] Refusal behavior
* [ ] Input filtering
* [ ] Output filtering
* [ ] Jailbreak testing
* [ ] Prompt-injection testing
* [ ] Abuse monitoring

---

# 31. Model Export

Model packages should contain:

```text
model/
├── weights
├── config
├── tokenizer
├── special_tokens
├── generation_config
└── metadata
```

Export targets:

* [x] Native checkpoint, BF16/FP16 loading, and CPU dynamic INT8
* [ ] INT4
* [ ] GGUF
* [x] SafeTensors and ONNX
* [ ] Runtime-specific formats

---

# 32. Serving

The serving layer should provide:

```text
                API Gateway
                     │
                     ▼
              Request Router
                     │
                     ▼
              Model Runtime
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       KV Cache   Batching   Scheduler
          │          │          │
          └──────────┼──────────┘
                     ▼
                  Response
```

APIs:

```text
POST /v1/chat/completions
POST /v1/completions
POST /v1/embeddings
GET  /v1/models
GET  /health
GET  /metrics
```

Features:

* [x] REST, OpenAI-compatible API, SSE, and WebSocket streaming
* [x] API-key authentication, rate limiting, cancellation, timeouts, and request queueing
* [x] Dynamic micro-batching
* [ ] Continuous batching
* [ ] Model routing

---

# 33. Production Infrastructure

Production runtime should support:

* [x] Docker, Docker Compose, GPU container profile, health, readiness, and liveness checks
* [ ] Load balancing
* [ ] Reverse proxy
* [ ] TLS
* [x] Authentication
* [ ] Authorization
* [x] Rate limiting
* [ ] Request tracing
* [x] Metrics and logging
* [ ] Model version routing

---

# 34. Observability

## Training metrics

```text
Loss
Cross Entropy
Perplexity
Learning Rate
Gradient Norm
Tokens/sec
Samples/sec
GPU utilization
GPU memory
CPU utilization
```

## Inference metrics

```text
Requests/sec
Tokens/sec
Time To First Token
Time Per Output Token
P50 latency
P95 latency
P99 latency
Queue latency
GPU utilization
GPU memory
KV cache usage
Error rate
```

## Logging

Every request should have a correlation/request ID.

---

# 35. Reproducibility

Every experiment should record:

```text
Model version
Git commit
Tokenizer version
Dataset version
Dataset mixture
Random seed
CUDA version
Framework version
GPU
Driver
Batch size
Gradient accumulation
Sequence length
Optimizer
Learning rate
Scheduler
Checkpoint
```

Example:

```json
{
  "model_version": "0.1.0",
  "tokenizer_version": "0.1.0",
  "dataset_version": "v3",
  "git_commit": "abc123",
  "seed": 42,
  "batch_size": 2,
  "gradient_accumulation": 16,
  "sequence_length": 1024
}
```

---

# 36. Fault Tolerance

Training and serving should handle failures safely.

Features:

* [x] Atomic checkpoints and checkpoint validation
* [ ] Automatic recovery
* [x] Resume training and training-state recovery
* [ ] Worker failure recovery
* [ ] Process restart
* [ ] GPU failure detection
* [ ] Corrupt checkpoint detection
* [ ] Training state recovery

---

# 37. Model Registry

Model registry:

```text
models/
├── base/
│   ├── v0.1
│   └── v0.2
│
├── instruct/
│   └── v0.1
│
├── reasoning/
│   └── v0.1
│
└── domain/
    └── v0.1
```

Metadata:

```text
Version
Architecture
Tokenizer
Dataset
Training config
Checkpoint
Evaluation
Quantization
Deployment status
```

---

# 38. Security

Production LLM infrastructure should include:

* [x] API-key authentication and rate limiting
* [~] Allowlisted tool authorization; no general RBAC system
* [ ] Request quotas
* [ ] Input validation
* [ ] Output validation
* [x] Deterministic prompt-injection guardrail probes and tool allowlists
* [x] Workspace path boundary and allowlisted test/git operations
* [ ] Secret isolation
* [ ] Audit logs

Tools should never automatically receive unrestricted system access.

---

# 39. Testing

Testing should exist at every layer.

## Unit tests

```text
Tokenizer
Attention
RoPE
RMSNorm
SwiGLU
Loss
Optimizer
Scheduler
Sampler
KV cache
```

## Integration tests

```text
Dataset → Training
Checkpoint → Resume
Model → Inference
Inference → WebSocket
LLM → Tool
RAG → LLM
Agent → Tool
```

## Regression tests

Maintain a fixed evaluation prompt set:

```text
"The cat is"
"Explain HTTP"
"Solve 25 × 37"
"Write a Python function..."
"Return valid JSON..."
"Summarize this document..."
```

Every new checkpoint can be compared against previous versions.

---

# 40. Project Architecture

A scalable project structure can evolve toward:

```text
llm-engine/
│
├── configs/
│   ├── model/
│   ├── training/
│   ├── inference/
│   ├── serving/
│   └── datasets/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── tokenized/
│   └── shards/
│
├── tokenizer/
│   ├── training/
│   ├── encoding/
│   └── decoding/
│
├── model/
│   ├── embeddings/
│   ├── attention/
│   ├── layers/
│   ├── normalization/
│   ├── positional/
│   ├── moe/
│   ├── multimodal/
│   └── transformer/
│
├── training/
│   ├── trainer/
│   ├── optimizers/
│   ├── schedulers/
│   ├── distributed/
│   ├── checkpoint/
│   └── callbacks/
│
├── finetuning/
│   ├── sft/
│   ├── lora/
│   ├── qlora/
│   ├── dpo/
│   └── preference/
│
├── reasoning/
│   ├── datasets/
│   ├── training/
│   ├── evaluation/
│   └── verification/
│
├── inference/
│   ├── runtime/
│   ├── generation/
│   ├── sampling/
│   ├── kv_cache/
│   ├── batching/
│   ├── quantization/
│   └── speculative/
│
├── rag/
│   ├── ingestion/
│   ├── chunking/
│   ├── embeddings/
│   ├── retrieval/
│   └── reranking/
│
├── tools/
│   ├── schemas/
│   ├── registry/
│   ├── executor/
│   └── permissions/
│
├── agents/
│   ├── planner/
│   ├── runtime/
│   ├── memory/
│   └── workflows/
│
├── multimodal/
│   ├── vision/
│   ├── audio/
│   └── processors/
│
├── evaluation/
│   ├── benchmarks/
│   ├── regression/
│   ├── reasoning/
│   ├── safety/
│   └── reports/
│
├── serving/
│   ├── http/
│   ├── websocket/
│   ├── streaming/
│   ├── scheduler/
│   └── middleware/
│
├── export/
│   ├── native/
│   ├── gguf/
│   ├── onnx/
│   └── quantized/
│
├── observability/
│   ├── logging/
│   ├── metrics/
│   ├── tracing/
│   └── profiling/
│
├── safety/
│   ├── filters/
│   ├── policies/
│   └── evaluation/
│
├── registry/
│   ├── models/
│   ├── datasets/
│   └── experiments/
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   ├── export.py
│   └── benchmark.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── regression/
│
├── reports/
│
├── checkpoints/
│
├── docs/
│
└── README.md
```

---

# 41. Development Roadmap

## Phase 1 — Foundation

### Status

* [x] Transformer
* [x] Tokenizer
* [x] Pretraining
* [x] Dataset loading
* [x] Validation
* [x] Checkpointing
* [x] Resume
* [x] GPU training
* [x] DDP foundation
* [x] Generation
* [x] WebSocket foundation

---

## Phase 2 — Training Quality

### Priority: Critical

* [x] Dataset cleaning, deduplication, PII filtering, governance, and packing
* [x] Learning-rate scheduling, warmup, gradient clipping, mixed precision, and non-finite checks
* [x] Training reports and reproducibility metadata
* [ ] Advanced curriculum and gradient diagnostics

---

## Phase 3 — Evaluation

### Priority: Critical

* [x] Automated benchmark runner, perplexity, domain evaluation, and regression tests
* [ ] Reasoning evaluation
* [ ] Code evaluation
* [ ] Instruction evaluation
* [ ] Long-context evaluation
* [x] Versioned deterministic regression matrix
* [ ] Model comparison reports

---

## Phase 4 — Instruction Model

### Priority: Critical

* [x] Chat template, instruction data, SFT, multi-turn conversations, and system prompts
* [ ] Structured output
* [ ] Better sampling

Result:

```text
Base LLM
   ↓
SFT
   ↓
Instruction/Chat LLM
```

---

## Phase 5 — Efficient Inference

### Priority: Critical

* [x] KV cache, paged KV accounting, prefix caching, and dynamic batching
* [ ] FlashAttention
* [x] CPU INT8 quantization and SafeTensors/ONNX export
* [ ] Continuous batching and full paged-attention execution

---

## Phase 6 — Reasoning

### Priority: High

* [ ] Math data
* [ ] Code data
* [ ] Reasoning data
* [ ] Reasoning SFT
* [ ] Verification
* [x] DPO
* [ ] Reasoning evaluation

---

## Phase 7 — Tools

### Priority: High

* [x] Tool schemas, structured arguments, local/MCP execution, and result handling
* [ ] Tool permissions
* [ ] Tool-use training

---

## Phase 8 — RAG

### Priority: High

* [x] Document ingestion, chunking, local retrieval, and RAG prompt construction
* [ ] Reranking
* [ ] Context construction
* [ ] Citations

---

## Phase 9 — Agents

### Priority: Medium/High

* [ ] Planner
* [ ] Agent loop
* [x] Bounded sequential tool orchestration, structured error recovery, and scoped session memory
* [~] Multi-step workflows without autonomous planning/reflection
* [ ] Human approval

---

## Phase 10 — Long Context

### Priority: Medium

```text
1K
 ↓
2K
 ↓
4K
 ↓
8K
 ↓
16K+
```

Add:

* [x] Opt-in NTK and YaRN RoPE scaling policies
* [ ] Long-context training
* [ ] Long-context evaluation
* [ ] Efficient attention
* [ ] KV optimization

---

## Phase 11 — Advanced Inference

### Priority: Medium

* [ ] MTP
* [ ] Draft model
* [ ] Speculative decoding
* [ ] Advanced batching
* [ ] INT4 KV cache

---

## Phase 12 — Advanced Architecture

### Priority: Research

* [ ] Linear attention
* [ ] Hybrid attention
* [ ] Sliding-window attention
* [ ] Sparse attention
* [x] Configurable sparse MoE and expert routing
* [ ] Expert parallelism

---

## Phase 13 — Multimodal

### Priority: Research/Product dependent

* [x] Vision encoder, projector, and visual tokens (small adapter profile)
* [ ] Multimodal datasets
* [ ] Multimodal SFT
* [ ] OCR
* [ ] Audio
* [ ] Video

---

## Phase 14 — Extreme Optimization

### Priority: Research

* [ ] 2-bit quantization
* [ ] 1-bit/ternary weights
* [ ] Custom CUDA kernels
* [ ] Tensor parallelism
* [ ] Pipeline parallelism
* [ ] Expert parallelism

---

# 42. Priority Matrix

| Feature              | Priority | Required?                |
| -------------------- | -------- | ------------------------ |
| Transformer          | P0       | Yes                      |
| Tokenizer            | P0       | Yes                      |
| Dataset pipeline     | P0       | Yes                      |
| Pretraining          | P0       | Yes                      |
| Evaluation           | P0       | Yes                      |
| Checkpoint/resume    | P0       | Yes                      |
| Reproducibility      | P0       | Yes                      |
| Instruction SFT      | P0       | For chat                 |
| KV cache             | P0       | For efficient generation |
| Inference            | P0       | Yes                      |
| Quantization         | P1       | Production               |
| Structured output    | P1       | Production               |
| Tool calling         | P1       | Agent capability         |
| RAG                  | P1       | Knowledge applications   |
| Embeddings           | P1       | RAG/search               |
| DPO                  | P1       | Alignment                |
| Reasoning            | P1       | Advanced capability      |
| FlashAttention       | P1       | Performance              |
| Long context         | P2       | Useful                   |
| Continuous batching  | P2       | Production scale         |
| Speculative decoding | P2       | Performance              |
| Hybrid attention     | P2       | Long-context efficiency  |
| Linear attention     | P2       | Advanced                 |
| Agent runtime        | P2       | Product capability       |
| Vision               | P2/P3    | Product dependent        |
| MoE                  | P3       | Research/scale           |
| 1-bit model          | P3       | Research                 |
| Audio                | P3       | Product dependent        |
| Video                | P3       | Research/product         |
| Custom CUDA kernels  | P3       | Optimization             |
| Tensor parallelism   | P3       | Large-scale training     |

---

# 43. Hardware Strategy

The current development environment is based around a GPU with approximately 4 GB VRAM.

Therefore the project should prioritize:

```text
Memory efficiency
       >
Data quality
       >
Training stability
       >
Inference optimization
       >
Architecture efficiency
       >
Parameter count
```

Recommended techniques:

```text
Gradient Accumulation
Gradient Checkpointing
Mixed Precision
Small Batch Size
Efficient Data Loading
LoRA/QLoRA
Quantized Inference
CPU Offloading
```

Avoid increasing the model size aggressively until the training and evaluation pipeline is stable.

---

# 44. Target Architecture

The long-term target:

```text
                              USER
                                │
                                ▼
                        ┌───────────────┐
                        │  API Gateway  │
                        └───────┬───────┘
                                │
                                ▼
                        ┌───────────────┐
                        │ Request Router│
                        └───────┬───────┘
                                │
             ┌──────────────────┼──────────────────┐
             │                  │                  │
             ▼                  ▼                  ▼
            RAG               Tools              Memory
             │                  │                  │
             └──────────────────┼──────────────────┘
                                │
                                ▼
                      ┌──────────────────┐
                      │   LLM Runtime    │
                      │                  │
                      │  Transformer     │
                      │  GQA             │
                      │  RoPE            │
                      │  RMSNorm         │
                      │  SwiGLU          │
                      │  KV Cache        │
                      │  Hybrid Attention│
                      └────────┬─────────┘
                               │
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
          Text              Tool Call        Structured
          Output                              Output
             │
             ▼
       ┌───────────────┐
       │ Safety Layer  │
       └───────┬───────┘
               │
               ▼
             USER
```

---

# 45. Final Goal

The final system should become a complete LLM platform rather than only a Transformer implementation.

```text
                         LLM PLATFORM
                              │
       ┌──────────────────────┼──────────────────────┐
       │                      │                      │
       ▼                      ▼                      ▼
   DATA SYSTEM           MODEL SYSTEM          TRAINING SYSTEM
       │                      │                      │
       │                      │                      │
       ▼                      ▼                      ▼
  Datasets              Transformer             Pretraining
  Cleaning              Attention               SFT
  Deduplication         GQA                     DPO
  Filtering             RoPE                    Reasoning
  Tokenization          SwiGLU                  Fine-tuning
       │                KV Cache                     │
       └──────────────────────┼──────────────────────┘
                              │
                              ▼
                         MODEL REGISTRY
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
             Base          Instruct       Reasoning
             Model           Model          Model
               │              │              │
               └──────────────┼──────────────┘
                              ▼
                        INFERENCE ENGINE
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
             Cache        Quantization    Batching
               │              │              │
               └──────────────┼──────────────┘
                              ▼
                       CAPABILITY LAYER
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
            RAG              Tools            Vision
             │                │                │
             └────────────────┼────────────────┘
                              ▼
                         AGENT RUNTIME
                              │
                              ▼
                        PRODUCTION API
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
          Observability     Security       Monitoring
```

---

# Definition of Done

The project should be considered a **complete production-oriented LLM platform** when it can:

```text
✓ Train a foundation model
✓ Process and version datasets
✓ Train reproducibly
✓ Resume failed training
✓ Evaluate checkpoints automatically
✓ Fine-tune with SFT
✓ Produce instruction-following responses
✓ Perform reasoning tasks
✓ Support preference optimization
✓ Maintain an efficient KV cache
✓ Generate streaming responses
✓ Produce structured JSON
✓ Call external tools
✓ Execute multi-step tool workflows
✓ Perform RAG
✓ Generate embeddings
✓ Rerank retrieved documents
✓ Maintain application-level memory
✓ Support longer contexts
✓ Quantize models
✓ Export models
✓ Run batched inference
✓ Expose production APIs
✓ Monitor latency and throughput
✓ Track GPU/CPU resources
✓ Authenticate API requests
✓ Apply tool permissions
✓ Run safety evaluations
✓ Maintain model versions
✓ Reproduce experiments
✓ Recover from failures
```

Advanced capabilities can then be added:

```text
→ Speculative decoding
→ MTP
→ Hybrid attention
→ Linear attention
→ MoE
→ Vision
→ Audio
→ Video
→ 1-bit models
→ Custom CUDA kernels
→ Large-scale distributed training
```

---

# Philosophy

The project follows one important principle:

> **Build a complete small LLM before building an incomplete giant LLM.**

The priority is not to copy every feature from a 27B+ model.

Instead:

```text
Reliable foundation
        ↓
Better data
        ↓
Better training
        ↓
Better evaluation
        ↓
Instruction following
        ↓
Reasoning
        ↓
Efficient inference
        ↓
Tools
        ↓
RAG
        ↓
Agents
        ↓
Long context
        ↓
Advanced architecture
        ↓
Multimodal capabilities
```

This approach allows the project to remain useful on limited hardware while providing a path toward larger, more capable models as hardware and datasets scale.

---

# License

Add the project's chosen license here.

Example:

```text
MIT License
```

---

# Project Status

**Current stage:** Foundation LLM / Research & Development

**Primary objective:** Build a complete, modular LLM engine from the ground up.

**Long-term objective:** A production-ready platform covering the complete lifecycle of modern LLM development — from raw data and pretraining through alignment, reasoning, tools, RAG, agents, efficient inference, multimodality, and production deployment.
