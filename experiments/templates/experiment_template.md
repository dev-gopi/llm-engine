# Experiment Manifest: EXP-{{ID}} — {{TITLE}}

## 1. Metadata
- **Experiment ID**: `EXP-{{ID}}`
- **Date**: YYYY-MM-DD
- **Author / Agent**: {{AUTHOR}}
- **Status**: [ ACTIVE | COMPLETED | ABORTED ]
- **Git Commit**: `{{GIT_COMMIT}}`
- **Primary Configuration**: `configs/{{CONFIG_FILE}}.yaml`

---

## 2. Hypothesis & Objectives
- **Hypothesis**: What specific improvement or capability is being tested?
- **Success Criteria**:
  - Validation loss $\le X.XX$
  - Validation perplexity $\le X.XX$
  - Zero degradation on retention benchmark.

---

## 3. Training & Hardware Environment
- **Hardware**: NVIDIA GeForce RTX 3050 (4 GB VRAM)
- **Model Architecture**:
  - Layers: 16
  - Hidden Size: 512
  - Heads: 8 / KV Heads: 2
  - Vocab: 40,000 / 42,000
- **Dataset**:
  - Mixture: {{DATASET_MIXTURE}}
  - Sequence Length: 512
  - Effective Batch Size: 32 (batch size 2 * accumulation 16)
- **Optimizer**: AdamW (`lr=5e-5`, `cosine`, `weight_decay=0.1`, `ema=0.999`)

---

## 4. Results & Metrics

| Metric | Baseline Value | Experiment Value | Delta |
| :--- | :--- | :--- | :--- |
| **Final Train Loss** | | | |
| **Validation Loss** | | | |
| **Validation Perplexity** | | | |
| **Retention Loss** | | | |
| **Peak VRAM Use** | | | |
| **Tokens / Sec (TPS)** | | | |

---

## 5. Observations & Qualitative Generation Samples

### Prompt 1:
> {{PROMPT}}

**Generated Output**:
> {{OUTPUT}}

---

## 6. Decision & Next Steps
- **Decision**: [ PROMOTE TO BEST CHECKPOINT | REJECT | RETRAIN ]
- **Rationale**:
- **Follow-up Tasks**:

