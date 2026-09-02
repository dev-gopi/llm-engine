# V2 capabilities, purposes, and scaling guide

This document describes what the current v2 model can realistically do, where
it can be useful, what it cannot safely do, and how this project can grow from
the active 54.4M-parameter model toward the supplied 1B, 7B, and 30B
architecture targets.

- To build and train the model, read
  [V2_TRAINING_GUIDE.md](V2_TRAINING_GUIDE.md).
- To generate, chat, serve an API, or use the browser UI, read
  [V2_USAGE_GUIDE.md](V2_USAGE_GUIDE.md).

## Current model

The active GPU model is defined by `configs/model.gpu.yaml`:

| Property | Current value |
| --- | ---: |
| Parameters | 54,405,632 (54.4M) |
| Vocabulary | 32,000 base tokens |
| Hidden size | 512 |
| Transformer layers | 10 |
| Attention heads | 8 query heads, 2 KV heads |
| Feed-forward size | 2,048 |
| Maximum context | 512 tokens |
| Position encoding | RoPE |
| Normalization | RMSNorm |
| Feed-forward activation | SwiGLU |
| Input/output embeddings | Tied |
| Activation checkpointing | Enabled |
| FP16/BF16 weight size | About 0.10 GiB |
| FP32 weight size | About 0.20 GiB |

Training uses much more memory than weights alone because it also needs
gradients, optimizer state, activations, temporary buffers, and—outside FSDP—an
EMA copy.

## Training stages and what they add

The same architecture behaves differently after each stage:

| Checkpoint | Main capability | Not yet expected |
| --- | --- | --- |
| `v2-pretraining/best.pt` | English text continuation, short stories, basic general-language patterns | Reliable assistant/chat behavior |
| `v2-pretraining-continued/best.pt` | More WikiText-weighted general and factual language patterns | Instruction following |
| `v2-finetuning/best.pt` | Chat, instructions, Bengali, Hindi, basic coding, GSM8K-style arithmetic, safety responses | Strong preference alignment |
| `v2-dpo/best.pt` | Better preference toward helpful/coherent responses from the available HelpSteer pairs | New factual knowledge not present in prior stages |

DPO primarily changes which response style the model prefers. It is not a
replacement for pretraining data, factual data, reasoning data, or SFT.

## What the trained model can do

Actual quality must be measured from the resulting checkpoint. With successful
pretraining, SFT, and DPO, the project is designed for the following bounded
capabilities.

### Text and story generation

- Continue short English prompts.
- Generate TinyStories-style simple stories.
- Produce short summaries, rewrites, lists, and explanations.
- Generate creative text with sampling controls.

The model is most likely to be coherent on short outputs similar to its
training distribution.

### Basic assistant chat

- Maintain a short conversation within its context window.
- Follow simple, direct instructions.
- Use a configurable system prompt and Gopi identity.
- Return plain text or Markdown-oriented responses.
- Produce safer refusals for examples represented in alignment data.

### Multilingual text

- Read and generate examples in English, Bengali, Hindi, and Hinglish.
- Understand common emoji and some complex emoji sequences.
- Answer basic multilingual questions represented by the SFT datasets.

Multilingual quality will be much smaller and less robust than a large model
trained on billions of high-quality multilingual tokens.

### Basic coding

- Answer simple Python syntax questions.
- Generate short functions and code snippets.
- Explain elementary programming concepts.
- Work with common instruction formats represented in CodeAlpaca and code
  instruction data.

Always execute generated code in an isolated environment and review it. The
model is not a compiler, security auditor, or reliable source of production
code.

### Basic arithmetic and reasoning

- Solve short arithmetic word problems similar to GSM8K examples.
- Perform simple step-by-step transformations.
- Answer basic factual and reading-comprehension questions.

The model should not be expected to perform dependable multi-step reasoning,
advanced mathematics, or precise long calculations. Use the API calculator
tool for deterministic arithmetic.

### Local application integration

- One-shot CLI generation.
- Interactive terminal chat.
- Browser-based chat UI.
- REST and WebSocket generation APIs.
- Conversation sessions stored in SQLite.
- Optional calculator and date/time tools.
- Optional web-search context through SearXNG or Brave.
- Allowlisted MCP tool calls.
- SafeTensors, `torch.export`, and ONNX export paths.

Web search and tools are external runtime features. They do not become part of
the model's learned weights.

## Useful project purposes

The current model is appropriate for:

- learning how an LLM tokenizer, transformer, trainer, evaluator, and server
  work end to end;
- testing dataset preparation and governance workflows;
- experimenting with pretraining, SFT, DPO, EMA, mixed precision, and
  checkpoint recovery;
- building a small local chatbot or browser demo;
- prototyping multilingual or domain-specific data mixtures;
- testing API, streaming, session, caching, and tool integrations;
- benchmarking inference and training optimizations on limited hardware;
- serving as a base for research and educational experiments.

It is not appropriate as the sole system for medical, legal, financial,
safety-critical, or other high-stakes decisions.

## Important limitations

### Small parameter count

54.4M parameters are useful for learning and bounded tasks but far smaller than
modern general-purpose assistants. The model has limited capacity for facts,
reasoning, languages, and instruction diversity.

### Short context

The current architecture supports 512 tokens. The usable prompt plus generated
response must fit within that window. It cannot reliably process large files,
books, or long conversations without chunking and retrieval.

### Hallucination

The model can produce fluent but false information. DPO does not guarantee
truthfulness. Validate factual output against trusted sources.

### No native multimodality

The current model accepts text token IDs. It does not directly understand
images, audio, or video. Adding those capabilities requires encoders, adapters,
multimodal data, objectives, and inference changes—not only more text data.

### Knowledge is bounded by data

The checkpoint does not automatically know current events or information never
represented by its training data. Runtime web search can provide current text
context, but the model must still interpret it correctly.

### Evaluation is required

Low aggregate validation loss does not prove high assistant quality. Evaluate
English, Bengali, Hindi, coding, GSM8K, chat, safety, repetition, and factual
accuracy independently, then inspect human-reviewed generations.

## How large this project can grow

The model implementation is configuration-driven and already includes future
architecture examples. The numbers below come from `scripts/inspect_model.py`
and describe weights and one maximum-length BF16 KV cache—not complete training
memory.

| Profile | Parameters | Context | FP16/BF16 weights | FP32 weights | BF16 KV cache per max-length sequence |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current v2 GPU | 54.4M | 512 | 0.10 GiB | 0.20 GiB | 0.002 GiB |
| Future 1B | 1.148B | 8,192 | 2.14 GiB | 4.28 GiB | 0.375 GiB |
| Future 7B | 6.511B | 32,768 | 12.13 GiB | 24.26 GiB | 4.50 GiB |
| Future 30B | 30.128B | 32,768 | 56.12 GiB | 112.24 GiB | 4.13 GiB |

The 1B, 7B, and 30B files are architecture targets, not proof that those models
can be trained well with the current datasets or laptop hardware.

## Practical scaling path

### Stage 1: improve the existing 54.4M model

Before increasing parameters:

1. Finish base pretraining and retain the validation-selected checkpoint.
2. Measure every domain independently.
3. Remove duplicates, leakage, low-quality answers, and privacy-sensitive data.
4. Increase factual, Bengali, Hindi, reasoning, and code quality—not only row
   count.
5. Run SFT and DPO with held-out evaluation and human review.
6. Use packed token shards when runtime tokenization becomes a bottleneck.

A well-trained small model is more useful than a larger model trained on an
unreviewed or badly mixed corpus.

### Stage 2: intermediate experiments

Before jumping to 1B, create intermediate configurations such as 100M–350M.
This tests whether quality improves with capacity while keeping failures and
cost manageable. Preserve the tokenizer mapping or intentionally begin a new
model family.

Increase model dimensions only after confirming:

- enough unique high-quality tokens;
- stable mixed-precision gradients;
- checkpoint and resume reliability;
- meaningful validation improvements;
- acceptable training time and storage;
- inference memory for the intended context and concurrency.

### Stage 3: approximately 1B parameters

`configs/text/model.future.1b.yaml` defines a 1.185B model with an 8K context and
the 50K from-scratch multilingual vocabulary from
`configs/text/tokenizer.future.50k.yaml`. The repository includes
`configs/text/pretraining.future.fsdp.yaml` as an FSDP example.
This stage requires substantially more unique data, compute, storage, and
validation than the laptop profile. Multi-GPU BF16 training, distributed
checkpoints, binary token shards, and cluster validation are the expected path.

### Stage 4: 7B and 30B targets

The 7B and 30B configurations require serious multi-GPU or multi-node
resources. Weight memory is only the beginning: training also holds gradients,
optimizer states, activations, communication buffers, and checkpoints. Long
contexts add large KV-cache and activation costs. Plan these stages as cluster
projects, not upgrades for a 4 GB laptop GPU.

## Data growth requirements

Increasing parameters without increasing unique, clean data usually produces a
larger model that memorizes the same corpus. As model size grows:

- use substantially more unique pretraining tokens;
- balance languages and domains intentionally;
- keep train, validation, and test examples disjoint;
- record source, version, license, allowed use, and privacy review;
- deduplicate both exact and near-duplicate content;
- filter secrets, personal data, corrupted text, and low-quality synthetic
  answers;
- reserve stable evaluation sets that are never used for tokenizer training,
  pretraining, SFT, or DPO;
- track tokens seen, not only examples or epochs.

Repeated epochs over a small corpus cannot replace new information.

## Engineering work required for larger models

The repository already contains many building blocks, but larger runs still
require target-hardware validation and operational work:

- FSDP or hybrid sharding across real GPU nodes;
- NCCL and network validation;
- reshardable distributed checkpoints;
- packed, memory-mapped token datasets;
- fault-tolerant job scheduling and checkpoint storage;
- throughput, utilization, memory, and communication profiling;
- larger-scale evaluation and contamination checks;
- inference quantization or tensor parallelism;
- production authentication, TLS, monitoring, rate limits, and abuse controls.

Run the planner before allocating a larger model:

```bash
.venv/bin/python scripts/inspect_model.py configs/text/model.future.1b.yaml

.venv/bin/python scripts/plan_training.py \
  --model-config configs/text/model.future.1b.yaml \
  --training-config configs/text/pretraining.future.fsdp.yaml \
  --training-tokens 20000000000 \
  --hardware-tflops 100 \
  --utilization 0.35 \
  --gpu-memory-gib 24 \
  --require-fit
```

The `training-tokens` value above is an illustrative planning input, not a
guarantee of quality or a final data recommendation. Replace performance and
memory inputs with measured values from the target hardware.

## Recommended direction for this project

For the current RTX 3050 laptop workflow:

```text
54.4M base pretraining
        -> optional one-epoch WikiText-heavy continuation
        -> quality-balanced multilingual SFT
        -> HelpSteer DPO
        -> domain evaluation and human review
        -> local CLI/UI/API deployment
```

Improve the 54.4M model and its data first. Then test one intermediate model
before attempting the provided 1B distributed profile. The 7B and 30B targets
should remain future cluster-scale experiments until data, evaluation,
distributed reliability, and compute budgets are ready.
