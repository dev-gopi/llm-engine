# Evaluation & Regression Testing (`docs/EVALUATION.md`)

*Authoritative Source: [`src/training/evaluator.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/training/evaluator.py), [`src/evaluation/benchmarks.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/evaluation/benchmarks.py), [`scripts/evaluate.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/scripts/evaluate.py)*

---

## 1. Metrics & Evaluation Methodology

Loss and perplexity are necessary for monitoring pretraining convergence, but complete evaluation requires multi-domain tracking:

1. **Validation Cross-Entropy**:
   $$\text{Loss} = -\frac{1}{N} \sum_{i=1}^N \log P(x_i \mid x_{<i})$$

2. **Perplexity (PPL)**:
   $$\text{PPL} = \exp(\text{Loss})$$

3. **Domain-Weighted Validation**: Evaluator weights held-out sets according to explicit capability matrices (`validation_domains` in `configs/pretraining.gpu.yaml`).

4. **Retention Loss Gating**: During fine-tuning, evaluator calculates retention loss on pretraining hold-outs to ensure fine-tuning does not destroy base capabilities (catastrophic forgetting).

---

## 2. Running Evaluation

### Standard Evaluation:
```bash
.venv/bin/python scripts/evaluate.py \
  --checkpoint checkpoints/finetuning/best.pt \
  --config configs/evaluation.finetuning.yaml
```

### Domain-Specific Evaluation:
```bash
.venv/bin/python scripts/evaluate_domains.py \
  --checkpoint checkpoints/finetuning/best.pt \
  --config configs/evaluation.domains.yaml
```

### Retention Regression Test:
```bash
.venv/bin/python scripts/evaluate_retention_loss.py \
  --checkpoint checkpoints/finetuning/best.pt \
  --baseline-report reports/pretraining.json
```

### Reasoning and Code Probes:
```bash
.venv/bin/python scripts/evaluate_benchmarks.py \
  --cases configs/evaluation.reasoning_code.jsonl \
  --checkpoint checkpoints/finetuning/best.pt
```

The bundled manifest is a small deterministic regression probe, not a
standardized benchmark or evidence of broad reasoning capability.

### Safety Guardrail Probes

`configs/evaluation.safety.jsonl` is a versioned, deterministic regression
manifest for prompt-injection, high-confidence harmful-request, and benign
control cases. It tests the pre-generation guardrail—not model alignment—and
is covered by `tests/test_prompt_safety.py`.

---

## 3. Automated Regression Tests

Regression contracts are guarded by automated tests in `tests/`:
- `tests/test_evaluation_regressions.py`: Verifies metric agreement and thresholds.
- `tests/test_retention_training.py`: Tests that chat tuning does not exceed retention degradation bounds.
- `tests/test_training_system.py`: Verifies early stopping and baseline checkpoint restoration.
