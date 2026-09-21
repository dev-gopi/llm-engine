# Current-State Snapshot (`docs/CURRENT_STATE.md`)

*Snapshot Date: September 2026*  
*Authoritative Verification: Verified against active source code and configuration files in `src/` and `configs/`.*

---

## 1. Model Architecture

| Property | Verified Value | Source File / Config |
| :--- | :--- | :--- |
| **Architecture Family** | Causal Decoder-Only Transformer | [`src/model/gpt.py`](../src/model/gpt.py) |
| **Active Base Parameter Count** | 81,314,304 parameters (~81.3M) | [`src/model/config.py`](../src/model/config.py), [`configs/model.gpu.yaml`](../configs/model.gpu.yaml) |
| **Generated Fine-Tuning Model Size** | ~82.3M parameters when the 42K append-only tokenizer artifact is loaded; the generated tokenizer directory is intentionally gitignored and is not present in a source-only ZIP | [`configs/tokenizer.yaml`](../configs/tokenizer.yaml), [`src/model/vocabulary.py`](../src/model/vocabulary.py) |
| **Hidden Size (`dim`)** | `512` | `configs/model.gpu.yaml` |
| **Number of Layers** | `16` | `configs/model.gpu.yaml` |
| **Attention Query Heads** | `8` (head dimension: `512 / 8 = 64`) | `configs/model.gpu.yaml` |
| **Key-Value Heads (`kv_heads`)** | `2` (Grouped-Query Attention, 4:1 query-to-KV ratio) | `configs/model.gpu.yaml` |
| **FFN Intermediate Dimension** | `2048` (`ffn_multiple_of: 128`) | `configs/model.gpu.yaml` |
| **FFN Activation** | `swiglu` (gated linear unit with SiLU activation) | [`src/model/feed_forward.py`](../src/model/feed_forward.py) |
| **Positional Encoding** | Rotary (`rotary`, RoPE), with opt-in linear, NTK-aware, and YaRN scaling policies; active profile remains unscaled for checkpoint compatibility | [`src/model/positional.py`](../src/model/positional.py) |
| **Normalization** | Pre-Norm `rms_norm`, `norm_eps: 1e-5`, zero learnable bias | [`src/model/layer_norm.py`](../src/model/layer_norm.py) |
| **Weight Tying** | `tie_word_embeddings: true` (input embeddings shared with LM head) | `configs/model.gpu.yaml` |
| **Context Length (`max_position`)**| `1024` tokens max capacity; trained at `512` tokens | `configs/model.gpu.yaml` |
| **Weight Initialization** | Normal distribution with standard deviation `0.02` | `configs/model.gpu.yaml` |
| **Gradient Checkpointing** | Enabled (`true`) — recomputes layer activations during backward | `configs/model.gpu.yaml` |

---

## 2. Tokenizer

| Property | Verified Value | Source File / Config |
| :--- | :--- | :--- |
| **Tokenizer Type** | Byte-Pair Encoding (BPE) with UTF-8 byte fallback | [`src/tokenizer/bpe.py`](../src/tokenizer/bpe.py) |
| **Base Vocabulary Size** | `40,000` tokens | [`configs/tokenizer.yaml`](../configs/tokenizer.yaml) |
| **Fine-Tuning Vocabulary Target** | `42,000` tokens via append-only generated extension; verify the generated artifact before a checkpoint-backed run | [`configs/tokenizer.yaml`](../configs/tokenizer.yaml), [`scripts/tokenize.py`](../scripts/tokenize.py) |
| **Special Tokens** | `<|pad|>` (0), `<|unk|>` (1), `<|bos|>` (2), `<|eos|>` (3), `<|mask|>` (4) | `src/tokenizer/bpe.py` |
| **Chat & Control Tokens** | Extended chat tags plus append-only `<thinking>` / `</thinking>` reasoning tags; chat SFT masks non-assistant tokens | [`src/model/vocabulary.py`](../src/model/vocabulary.py), [`src/inference/chat_session.py`](../src/inference/chat_session.py) |
| **Training Method** | Iterative pair frequency counting with byte-level pre-tokenization regex | [`src/tokenizer/trainer.py`](../src/tokenizer/trainer.py) |
| **Loading Process** | JSON vocab + merges file with atomic hash and size verification | `src/tokenizer/decoder.py` |

---

## 3. Training Infrastructure

| Property | Verified Value | Source File / Config |
| :--- | :--- | :--- |
| **Datasets** | Pretraining: `WikiText-103` (90%) + `TinyStories` (10%)<br>SFT: Instruction pilot (`core_chat`, `code_instructions`, `recovery_sft`) | [`configs/pretraining.gpu.yaml`](../configs/pretraining.gpu.yaml)<br>[`configs/finetuning.gpu.yaml`](../configs/finetuning.gpu.yaml) |
| **Dataset Loading** | Streaming `lazy_dataset: true` with multi-worker prefetching | [`src/datasets/loader.py`](../src/datasets/loader.py) |
| **Batch Size** | Micro-batch size `2` per step; gradient accumulation `16` (effective batch `32`) | `configs/pretraining.gpu.yaml` |
| **Sequence Length** | `512` tokens per sequence during training | `configs/pretraining.gpu.yaml` |
| **Optimizer** | AdamW (`lr=5e-5`, `weight_decay=0.1`, `beta1=0.9`, `beta2=0.95`, `eps=1e-8`) | [`src/optim/adamw.py`](../src/optim/adamw.py) |
| **Learning Rate Schedule** | Cosine decay with 5% linear warmup, `min_lr_ratio: 0.1` | [`src/optim/scheduler.py`](../src/optim/scheduler.py) |
| **Weight Averaging** | Exponential Moving Average (EMA) with decay rate `0.999` | [`src/optim/ema.py`](../src/optim/ema.py) |
| **Mixed Precision** | `fp16` (GradScaler initial scale: 1024, growth interval: 20000) or `bf16` | `configs/pretraining.gpu.yaml` |
| **Loss Function** | Chunked cross-entropy with `z_loss_coefficient: 0.0001` to stabilize logits | [`src/model/loss.py`](../src/model/loss.py) |
| **Checkpointing** | Atomic single-file checkpoints plus checksum-validated distributed shards; resume restores optimizer/scheduler/scaler, RNG, and sampler metadata | [`src/training/checkpoint.py`](../src/training/checkpoint.py), [`src/training/distributed_checkpoint.py`](../src/training/distributed_checkpoint.py) |
| **Checkpoint Promotion** | Deterministic multi-axis selection with protected-regression gates and release evidence | [`src/training/promotion.py`](../src/training/promotion.py), [`src/release/pipeline.py`](../src/release/pipeline.py) |
| **Evaluation Strategy** | Per-domain validation loss, early stopping (patience: 3, delta: 0.001) | [`src/training/evaluator.py`](../src/training/evaluator.py) |
| **Logging** | Console, structured log file (`logs/`), live training report JSON/HTML | [`src/training/reporting.py`](../src/training/reporting.py) |

---

## 4. Hardware Constraints

- **Active Training GPU**: Single NVIDIA GeForce RTX 3050 (4,096 MiB VRAM).
- **Driver / CUDA**: PyTorch 2.4+ with CUDA runtime.
- **Host Resources**: x86_64 Linux, 16–32 GB System RAM, SSD storage.
- **VRAM Budget Allocation**:
  - Model Weights (FP16/BF16): ~165 MB
  - Optimizer State (FP32 master weights + momentum + variance): ~990 MB
  - EMA Weights (FP16/BF16): ~165 MB
  - Activation Footprint (with gradient checkpointing & batch size 2): ~600–900 MB
  - Scratch buffers / CUDA overhead: ~600 MB
  - Total Peak Footprint: **~2.8 – 3.2 GB**, fitting comfortably within the 4 GB hardware ceiling.

---

## 5. Serving & Inference Runtime

| Property | Verified Value | Source File / Config |
| :--- | :--- | :--- |
| **API Server** | FastAPI with Uvicorn ASGI backend | [`src/serving/api.py`](../src/serving/api.py) |
| **WebSocket** | Full-duplex WebSocket streaming at `/ws/generate` | [`src/serving/websocket.py`](../src/serving/websocket.py) |
| **Continuous Batching** | Iteration-level request admission/completion with cancellation-aware scheduling | [`src/serving/batching.py`](../src/serving/batching.py), [`src/serving/orchestration.py`](../src/serving/orchestration.py) |
| **KV Cache** | Tensor cache plus block-paged KV accounting and immutable LRU prefix-cache reuse | [`src/model/kv_cache.py`](../src/model/kv_cache.py)<br>[`src/inference/paged_kv_cache.py`](../src/inference/paged_kv_cache.py) |
| **Quantization** | Portable FP16/BF16 and packed INT4 export, optional dynamic CPU INT8 weights, and per-token scaled INT8 paged-KV storage; interoperable GPTQ/AWQ/GGUF conversion remains `QNT-002` | [`src/inference/quantization.py`](../src/inference/quantization.py), [`scripts/export.py`](../scripts/export.py) |
| **Sampling Methods** | Greedy, Temperature, Top-K, Top-P (Nucleus), Repetition Penalty | [`src/inference/sampler.py`](../src/inference/sampler.py) |
| **Tool Execution** | Local/MCP tools with strict JSON-envelope/schema validation, bounded sequential orchestration, and a separate approval-gated bounded agent state machine; autonomous model-generated planning/reflection and parallel execution are not claimed | [`src/inference/local_tools.py`](../src/inference/local_tools.py), [`src/inference/agent_runtime.py`](../src/inference/agent_runtime.py), [`src/serving/backend.py`](../src/serving/backend.py) |
| **Long-Term Memory** | Opt-in user-scoped semantic/episodic SQLite memory with embedding retrieval, TTL expiry, purge, and user deletion | [`src/inference/memory.py`](../src/inference/memory.py) |
| **Reasoning Budget** | `none` / `low` / `medium` / `high` runtime token budgets with reasoning-token usage accounting | [`src/runtime/reasoning.py`](../src/runtime/reasoning.py), [`src/api/usage.py`](../src/api/usage.py) |

---

## 9. Optional Modern Architecture & Deployment Features (2026-09-21)

These features were added **without modifying the default `configs/model.gpu.yaml` checkpoint contract**.

| Capability | Verified local state | Authoritative files |
| :--- | :--- | :--- |
| Hybrid dense/linear attention | Opt-in repeating layer schedules; reference causal linear attention uses fixed-size recurrent decode state | [`src/model/attention.py`](../src/model/attention.py), [`src/model/config.py`](../src/model/config.py), [`configs/model.hybrid.gpu.yaml`](../configs/model.hybrid.gpu.yaml) |
| Sparse MoE balancing | Differentiable router load-balancing auxiliary objective plus entropy/expert-load diagnostics; disabled unless `moe_aux_loss_weight > 0` | [`src/model/feed_forward.py`](../src/model/feed_forward.py), [`src/training/trainer.py`](../src/training/trainer.py) |
| Q1_0 research packing | Engine-native binary weight packing at 1.125 effective bits/weight with self-describing safetensors export | [`src/inference/quantization.py`](../src/inference/quantization.py), [`scripts/export.py`](../scripts/export.py) |
| GGUF model serving | Delegation to a local OpenAI-compatible runtime such as `llama-server`; Gopi retains its API/UI while specialized kernels remain external | [`src/serving/external_backend.py`](../src/serving/external_backend.py), [`scripts/serve_gguf.py`](../scripts/serve_gguf.py) |
| Deployment memory planning | Weight + KV + fixed linear-state estimates across BF16/INT8/INT4/Q1 combinations, exposed through API and CLI | [`src/runtime/resource_planner.py`](../src/runtime/resource_planner.py), [`scripts/plan_deployment.py`](../scripts/plan_deployment.py) |
| Runtime capability metadata | Reports attention pattern, dense/linear layer counts, MoE topology, MTP horizons and QK normalization | [`src/runtime/capabilities.py`](../src/runtime/capabilities.py) |
| Playground controls | Reasoning effort, architecture/context summary, cached/reasoning tokens and analytical memory footprint | [`ui/index.html`](../ui/index.html), [`ui/app.js`](../ui/app.js) |
| Training report | Architecture summary, deployment matrix, MTP loss and MoE auxiliary loss visibility | [`scripts/build_training_report.py`](../scripts/build_training_report.py), [`reports/training_report.html`](../reports/training_report.html) |

See [`MODERN_MODEL_FEATURES_2026.md`](MODERN_MODEL_FEATURES_2026.md) for the external research references and explicit non-claims.
