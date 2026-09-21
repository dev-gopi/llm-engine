# Testing Strategy & Regression Suites (`docs/TESTING.md`)

*Authoritative Source: [`pyproject.toml`](../pyproject.toml), [`tests/`](../tests)*

---

## 1. Testing Framework & Organization

`llm-engine` maintains over 830 automated tests executed with `pytest`. The test suite validates tensor mathematical contracts, memory allocation bounds, tokenizer round-tripping, checkpoint persistence, and API endpoints.

```text
tests/
├── fixtures/                         # Mock MCP servers, synthetic datasets
├── test_gpt.py                       # Model forward passes & output shapes
├── test_attention.py                 # Multi-head & Grouped-Query attention math
├── test_model_config.py              # Configuration validation and parameter sizing
├── test_feed_forward.py              # SwiGLU, GeGLU, and FFN linear projections
├── test_positional.py                # RoPE rotation math and position offsets
├── test_layer_norm.py                # RMSNorm and LayerNorm invariants
├── test_loss.py                      # Chunked cross-entropy and z-loss
├── test_tokenizer.py                 # BPE encoding, decoding, and UTF-8 bytes
├── test_vocabulary_compatibility.py  # Append-only vocabulary expansion rules
├── test_datasets.py                  # Dataloaders, samplers, and token sharding
├── test_dataset_governance.py        # PII, toxicity, and deduplication filters
├── test_training_system.py           # End-to-end training loops and resume state
├── test_optim.py                     # AdamW weight decay decoupling & schedulers
├── test_dpo_workflow.py              # Direct Preference Optimization loss contracts
├── test_generation.py                # Autoregressive generation & KV caching
├── test_inference_precision.py       # INT8 dynamic quantization accuracy
├── test_serving.py                   # FastAPI endpoints and dynamic batching
├── test_serving_orchestration.py     # Request queues and timeout handling
├── test_mcp_client.py                # Model Context Protocol client & server RPC
├── test_local_tools.py               # Sandboxed workspace tools
├── test_rag.py                       # Vector search indexing and retrieval
└── test_evaluation_regressions.py    # Retention benchmarks and regression guards
```

---

## 2. Test Execution Commands

### Run Full Test Suite:
```bash
.venv/bin/pytest -q
```

### Run Fast Targeted Subsystem Tests:
```bash
# Model architecture tests
.venv/bin/pytest tests/test_model_config.py tests/test_gpt.py tests/test_attention.py -q

# Tokenizer tests
.venv/bin/pytest tests/test_tokenizer.py tests/test_vocabulary_compatibility.py -q

# Training system tests
.venv/bin/pytest tests/test_training_system.py tests/test_training_integration.py -q

# Serving & API tests
.venv/bin/pytest tests/test_serving.py tests/test_serving_orchestration.py -q
```

### Validate the task registry
```bash
.venv/bin/python scripts/audit_task_registry.py
```

The audit rejects duplicate task IDs, heading/declared-ID mismatches, and
undefined task dependencies.

### Data-preparation dependency note
The Hugging Face/Arrow preparation tests require the declared `pyarrow`
dependency. If it is unavailable in an offline environment, report those tests
as not executed; do not count them as passing.

---

## 3. Mandatory Testing Rules for AI Agents

1. **Run Tests Before and After Modifying Code**: Always execute the relevant test module before editing to confirm baseline behavior, and re-run afterwards.
2. **Zero Regression Tolerance**: All executable repository tests must pass; source-only archives may intentionally skip generated-artifact checks, and dependency-gated tests must be reported explicitly rather than silently omitted. A change that breaks an existing test is unacceptable unless the contract itself was intentionally redesigned via an approved ADR.
3. **No Mocking of Numerical Tensors**: Transformer numerical tests must use real PyTorch tensors (on CPU or CUDA) to catch shape and gradient errors.

