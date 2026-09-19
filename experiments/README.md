# Model Experiment Tracking Registry (`experiments/`)

This directory provides structured tracking for all pretraining, fine-tuning, DPO, and architectural experiments.

---

## Directory Layout

```text
experiments/
├── README.md               # Tracking guide and lifecycle instructions
├── active/                 # Manifests for experiments currently training or evaluating
├── completed/              # Archived manifests for completed experiments
└── templates/
    └── experiment_template.md  # Standard markdown template for new runs
```

---

## Experiment Lifecycle

1. **Before Running Training**:
   - Copy `experiments/templates/experiment_template.md` to `experiments/active/EXP-<ID>-<name>.md`.
   - Record git commit hash, configuration file path, dataset versions, and hypotheses.
2. **During Training**:
   - Record training start time, initial loss, and observed GPU memory utilization.
3. **After Completion**:
   - Move manifest to `experiments/completed/`.
   - Record validation metrics (loss, perplexity, GSM8K accuracy, retention loss).
   - Document decisions and next steps.

