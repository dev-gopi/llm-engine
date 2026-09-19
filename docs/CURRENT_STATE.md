# Current-State Snapshot (`docs/CURRENT_STATE.md`)

*Snapshot Date: September 2026*  
*Authoritative Verification: Verified against active source code and configuration files in `src/` and `configs/`.*

---

## 1. Model Architecture

| Property | Verified Value | Source File / Config |
| :--- | :--- | :--- |
| **Architecture Family** | Causal Decoder-Only Transformer | [`src/model/gpt.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/model/gpt.py) |
| **Active Base Parameter Count** | 81,314,304 parameters (~81.3M) | [`src/model/config.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/model/config.py), [`configs/model.gpu.yaml`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/configs/model.gpu.yaml) |
| **Extended Vocab Parameter Count**| 82,338,304 parameters (~82.3M) | [`data/tokenizer-finetuning/`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/data/tokenizer-finetuning/) |
| **Hidden Size (`dim`)** | `512` | `configs/model.gpu.yaml` |
| **Number of Layers** | `16` | `configs/model.gpu.yaml` |
| **Attention Query Heads** | `8` (head dimension: `512 / 8 = 64`) | `configs/model.gpu.yaml` |
| **Key-Value Heads (`kv_heads`)** | `2` (Grouped-Query Attention, 4:1 query-to-KV ratio) | `configs/model.gpu.yaml` |
| **FFN Intermediate Dimension** | `2048` (`ffn_multiple_of: 128`) | `configs/model.gpu.yaml` |
| **FFN Activation** | `swiglu` (gated linear unit with SiLU activation) | [`src/model/feed_forward.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/model/feed_forward.py) |
| **Positional Encoding** | Rotary (`rotary`, RoPE), `rope_base: 10000.0`, `rope_scale: 1.0` | [`src/model/positional.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/model/positional.py) |
| **Normalization** | Pre-Norm `rms_norm`, `norm_eps: 1e-5`, zero learnable bias | [`src/model/layer_norm.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/model/layer_norm.py) |
| **Weight Tying** | `tie_word_embeddings: true` (input embeddings shared with LM head) | `configs/model.gpu.yaml` |
| **Context Length (`max_position`)**| `1024` tokens max capacity; trained at `512` tokens | `configs/model.gpu.yaml` |
| **Weight Initialization** | Normal distribution with standard deviation `0.02` | `configs/model.gpu.yaml` |
| **Gradient Checkpointing** | Enabled (`true`) — recomputes layer activations during backward | `configs/model.gpu.yaml` |

---

## 2. Tokenizer

| Property | Verified Value | Source File / Config |
| :--- | :--- | :--- |
| **Tokenizer Type** | Byte-Pair Encoding (BPE) with UTF-8 byte fallback | [`src/tokenizer/bpe.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/tokenizer/bpe.py) |
| **Base Vocabulary Size** | `40,000` tokens | [`configs/tokenizer.yaml`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/configs/tokenizer.yaml) |
| **Fine-Tuning Vocabulary Size**| `42,000` tokens (verified append-only extension) | [`data/tokenizer-finetuning/`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/data/tokenizer-finetuning/) |
| **Special Tokens** | `<|pad|>` (0), `<|unk|>` (1), `<|bos|>` (2), `<|eos|>` (3), `<|mask|>` (4) | `src/tokenizer/bpe.py` |
| **Chat & Control Tokens** | Extended in fine-tuning: `<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>` | `src/model/vocabulary.py` |
| **Training Method** | Iterative pair frequency counting with byte-level pre-tokenization regex | [`src/tokenizer/trainer.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/tokenizer/trainer.py) |
| **Loading Process** | JSON vocab + merges file with atomic hash and size verification | `src/tokenizer/decoder.py` |

---

## 3. Training Infrastructure

| Property | Verified Value | Source File / Config |
| :--- | :--- | :--- |
| **Datasets** | Pretraining: `WikiText-103` (90%) + `TinyStories` (10%)<br>SFT: Instruction pilot (`core_chat`, `code_instructions`, `recovery_sft`) | [`configs/pretraining.gpu.yaml`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/configs/pretraining.gpu.yaml)<br>[`configs/finetuning.gpu.yaml`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/configs/finetuning.gpu.yaml) |
| **Dataset Loading** | Streaming `lazy_dataset: true` with multi-worker prefetching | [`src/datasets/loader.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/datasets/loader.py) |
| **Batch Size** | Micro-batch size `2` per step; gradient accumulation `16` (effective batch `32`) | `configs/pretraining.gpu.yaml` |
| **Sequence Length** | `512` tokens per sequence during training | `configs/pretraining.gpu.yaml` |
| **Optimizer** | AdamW (`lr=5e-5`, `weight_decay=0.1`, `beta1=0.9`, `beta2=0.95`, `eps=1e-8`) | [`src/optim/adamw.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/optim/adamw.py) |
| **Learning Rate Schedule** | Cosine decay with 5% linear warmup, `min_lr_ratio: 0.1` | [`src/optim/scheduler.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/optim/scheduler.py) |
| **Weight Averaging** | Exponential Moving Average (EMA) with decay rate `0.999` | [`src/optim/ema.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/optim/ema.py) |
| **Mixed Precision** | `fp16` (GradScaler initial scale: 1024, growth interval: 20000) or `bf16` | `configs/pretraining.gpu.yaml` |
| **Loss Function** | Chunked cross-entropy with `z_loss_coefficient: 0.0001` to stabilize logits | [`src/model/loss.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/model/loss.py) |
| **Checkpointing** | Single-file atomic checkpoint serialization (`latest.pt`, `best.pt`) | [`src/training/checkpoint.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/training/checkpoint.py) |
| **Evaluation Strategy** | Per-domain validation loss, early stopping (patience: 3, delta: 0.001) | [`src/training/evaluator.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/training/evaluator.py) |
| **Logging** | Console, structured log file (`logs/`), live training report JSON/HTML | [`src/training/reporting.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/training/reporting.py) |

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
| **API Server** | FastAPI with Uvicorn ASGI backend | [`src/serving/api.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/serving/api.py) |
| **WebSocket** | Full-duplex WebSocket streaming at `/ws/generate` | [`src/serving/websocket.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/serving/websocket.py) |
| **Dynamic Batching** | Configurable micro-batch queue with timeout aggregation | [`src/serving/batching.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/serving/batching.py) |
| **KV Cache** | Tensor-based standard KV cache + Block-Paged KV cache | [`src/model/kv_cache.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/model/kv_cache.py)<br>[`src/inference/paged_kv_cache.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/inference/paged_kv_cache.py) |
| **Quantization** | Dynamic INT8 weight quantization | [`src/inference/quantization.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/inference/quantization.py) |
| **Sampling Methods** | Greedy, Temperature, Top-K, Top-P (Nucleus), Repetition Penalty | [`src/inference/sampler.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/inference/sampler.py) |
| **Tool Execution** | Local filesystem/command tools + MCP client integration | [`src/inference/local_tools.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/inference/local_tools.py)<br>[`src/mcp/client.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/mcp/client.py) |

