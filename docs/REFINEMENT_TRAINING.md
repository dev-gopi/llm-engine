# Curated refinement training

Use this stage after SFT or recovery SFT to repair observed failures and make
responses more consistent. It intentionally mixes 60% reviewed corrections
with 40% broad, cleaned recovery data, which helps prevent forgetting.

## 1. Add reviewed examples

Create these files. They must be disjoint: never place the same prompt or a
near-duplicate conversation in both splits.

```text
data/processed/refinement/train.jsonl
data/processed/refinement/validation.jsonl
```

Each line is one JSON object in the SFT chat format:

```json
{"id":"repair-001","source":"human_review","messages":[{"role":"user","content":"Explain RAM simply."},{"role":"assistant","content":"RAM is a computer's short-term working memory. Programs use it while they run; its contents are usually lost when power is off."}]}
```

Use exact corrections for real failures, clear target answers, and consistent
tone. Exclude private information, unverified facts, malformed JSON, duplicate
prompts, and low-quality answers.

## 2. Prepare the retention data and validate paths

```bash
.venv/bin/python scripts/prepare_recovery_sft.py --output data/processed/recovery_sft

.venv/bin/python scripts/audit_datasets.py \
  --training-config configs/finetuning.refinement.gpu.yaml --stage sft
```

## 3. Train a new refinement stage

Use `--init-from` from the best prior stage. `--resume` is only for an
interrupted run with exactly the same configuration and data files.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/train.py \
  --model-config configs/model.gpu.yaml \
  --training-config configs/finetuning.refinement.gpu.yaml \
  --tokenizer data/tokenizer \
  --init-from checkpoints/recovery/best.pt \
  --output checkpoints/refinement/latest.pt \
  --best-output checkpoints/refinement/best.pt \
  --log-file logs/refinement.log \
  --report-json reports/refinement.json
```

If recovery was skipped, replace the `--init-from` path with
`checkpoints/finetuning/best.pt`.

Select `checkpoints/refinement/best.pt` only after it improves both the held-out
refinement examples and fixed behavioral evaluation prompts.
