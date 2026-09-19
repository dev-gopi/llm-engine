# Dependencies & Environment Requirements (`docs/DEPENDENCIES.md`)

*Authoritative Source: [`pyproject.toml`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/pyproject.toml)*

---

## 1. Core Python Dependencies

| Package | Version Constraint | Why It Exists | Used By |
| :--- | :--- | :--- | :--- |
| `torch` | `>=2.4` | Core tensor compute, autograd, AMP, neural network layers | Entire codebase |
| `numpy` | `>=1.26` | Array operations, binary memory mapping for token shards | `src/datasets/`, `src/model/` |
| `pyyaml` | `>=6.0` | Hierarchical configuration file parsing and inheritance | `src/utils/config.py` |
| `fastapi` | `>=0.115` | Production REST API endpoints and routing | `src/serving/api.py` |
| `uvicorn[standard]` | `>=0.30` | High-performance ASGI server for HTTP & WebSockets | `src/serving/` |
| `safetensors` | `>=0.4` | Safe, zero-copy, non-pickling tensor serialization | `src/model/`, `scripts/export.py` |
| `pyarrow` | `>=15` | High-throughput columnar dataset reading and sharding | `src/datasets/` |
| `regex` | `>=2024.5.15` | UTF-8 compliant pre-tokenization regular expressions | `src/tokenizer/bpe.py` |
| `python-dotenv` | `>=1.0` | Environment variable management from `.env` | `src/utils/config.py` |

---

## 2. Optional Dependency Groups

### Development (`dev`)
```toml
dev = ["pytest>=8", "httpx>=0.27"]
```
- `pytest`: Automated test runner for the 670+ test suites.
- `httpx`: Asynchronous HTTP client used for testing FastAPI endpoints.

### Dataset Processing (`data`)
```toml
data = ["datasets>=2.19", "datatrove>=0.3"]
```
- `datasets`: Hugging Face dataset download utility for raw benchmarks (e.g. GSM8K, WikiText).
- `datatrove`: Large-scale deduplication and tokenization utilities.

### Model Export (`export`)
```toml
export = ["onnx>=1.16", "onnxruntime>=1.18"]
```
- Used for converting PyTorch checkpoints to ONNX graphs and validating ONNX Runtime inference.

### Document RAG (`rag`)
```toml
rag = ["pypdf>=5"]
```
- PDF extraction and parsing for local document retrieval in `src/inference/rag.py`.

### Vision & Multimodal (`images`)
```toml
images = ["pillow>=10"]
```
- Image decoding and preprocessing for vision encoders (`src/vision/`).

---

## 3. System & CUDA Requirements

- **Operating System**: Linux (Ubuntu 22.04 / 24.04 LTS recommended).
- **Python**: `>= 3.10` (tested with Python 3.12).
- **NVIDIA Driver**: `>= 535.xx`.
- **CUDA Toolkit**: CUDA 12.1+ / 12.4 supported by PyTorch 2.4+.
- **Host RAM**: 16 GB minimum (32 GB recommended for large dataset sharding).
- **Disk Storage**: SSD recommended for fast binary shard memory-mapping.

