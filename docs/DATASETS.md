# Dataset Pipelines & Governance (`docs/DATASETS.md`)

*Authoritative Source: [`src/datasets/loader.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/datasets/loader.py), [`src/datasets/token_shards.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/datasets/token_shards.py), [`src/datasets/filters.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/datasets/filters.py)*

---

## 1. Active Dataset Mixtures

### 1.1 Pretraining Mixture
Configured in [`configs/pretraining.gpu.yaml`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/configs/pretraining.gpu.yaml):
- **WikiText-103 (90%)**: High-entropy encyclopedia text providing broad vocabulary, syntax, and factual structures.
- **TinyStories (10%)**: Synthetically generated simple narrative stories providing clean grammar, dialogue structures, and consistent discourse coherence.

### 1.2 Fine-Tuning (SFT) Corpora
Located in [`data/processed/`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/data/processed/):
- `core_chat/`: Multi-turn foundational conversational prompts and responses.
- `code_instructions/`: Code generation and debugging instruction pairs.
- `recovery_sft/`: Domain recovery subsets (math, logic, English, Hindi, Bengali) to prevent catastrophic forgetting.

---

## 2. Memory-Mapped Binary Token Shards

To avoid high RAM usage during training on large datasets, `llm-engine` stores tokenized sequences as binary shards:
- Files are saved as `.bin` memory-mapped arrays of `uint16` or `uint32`.
- Accompanied by `.idx` files storing sequence boundaries.
- **Streaming Loader (`src/datasets/token_shards.py`)**: `ShardedDataset` memory-maps the binary files, sampling batches in $O(1)$ memory without loading files into host memory.

### Building Shards:
```bash
.venv/bin/python scripts/build_token_shards.py \
  --input data/processed/fineweb_edu/ \
  --output data/shards/fineweb_edu/ \
  --tokenizer data/tokenizer/ \
  --seq-len 512
```

---

## 3. Dataset Governance & Filtering

Every dataset must pass governance checks in [`src/datasets/filters.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/datasets/filters.py):
1. **Length Filtering**: Documents shorter than 15 tokens or longer than maximum context are pruned.
2. **Repetition & Deduplication**: Repetitive n-grams and duplicate documents are rejected.
3. **PII & Safety**: Redacts email addresses, phone numbers, and IP addresses.
4. **License Audit**: The dataset governance policy in `configs/pretraining.gpu.yaml` specifies license verification (`commercial_use: false`).

Verified by [`tests/test_dataset_governance.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/tests/test_dataset_governance.py) and [`tests/test_audit_data_quality.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/tests/test_audit_data_quality.py).

