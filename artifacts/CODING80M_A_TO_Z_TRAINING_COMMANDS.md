# GopiCoder-80M — A-to-Z Training Commands

This file is a command-first runbook for training the current ~81.8M-parameter Coding Agent from scratch.

> Run commands from the **llm-engine project root** unless a step says otherwise.
>
> The current training order is:
>
> `environment → data acquisition → data preparation → tokenizer → preflight → Stage 1 → Stage 1 continuation → Stage 2 → SFT → DPO → evaluation/report → serve`

---

## 0. Enter the project

```bash
cd /path/to/llm-engine
pwd
```

Confirm important folders exist:

```bash
ls
ls configs/coding80m
ls coding80m
```

---

## 1. Create and activate Python environment

Example with Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Upgrade packaging tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Install the project and training/data dependencies:

```bash
python -m pip install -e '.[dev,data]'
```

Install the dataset downloader dependencies:

```bash
python -m pip install -U datasets huggingface_hub pyyaml
```

Optional dependencies for The Stack v2 Software Heritage/S3 access:

```bash
python -m pip install boto3 'smart_open[s3]'
```

Check environment:

```bash
python --version
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())"
python -c "import datasets, pyarrow, yaml; print('datasets:', datasets.__version__); print('pyarrow:', pyarrow.__version__)"
```

GPU check:

```bash
nvidia-smi
```

---

## 2. Install the Coding80M training bundle

If the Coding80M training bundle is not already merged into the project:

```bash
unzip -o /path/to/gopi-coder-80m-training-config-sample-data-resumable.zip -d /tmp/gopi80m-training
```

Inspect it:

```bash
find /tmp/gopi80m-training -maxdepth 3 -type f | sort
```

Copy/merge the contents of the inner project folder into your `llm-engine` root.

After merging, verify:

```bash
ls coding80m
ls configs/coding80m
```

Expected important configs:

```bash
ls \
  configs/coding80m/model.yaml \
  configs/coding80m/tokenizer.yaml \
  configs/coding80m/pretraining.stage1.yaml \
  configs/coding80m/pretraining.stage2.yaml \
  configs/coding80m/finetuning.sft.yaml \
  configs/coding80m/dpo.yaml \
  configs/coding80m/inference.yaml
```

---

## 3. Install DPO mid-epoch resume support

Run once:

```bash
chmod +x coding80m/*.sh
./coding80m/install_resume_support.sh
```

This creates backups under:

```bash
find .resume-support-backups -maxdepth 3 -type f 2>/dev/null | tail
```

---

## 4. Install dataset downloader files

If these are not already in the project:

```bash
unzip -o /path/to/gopi80m-config-driven-dataset-downloader.zip -d /tmp/gopi80m-downloader
```

Copy:

```text
scripts/download_coding80m_datasets.py
configs/datasets.coding80m.yaml
```

into the project root at the same paths.

Verify:

```bash
ls scripts/download_coding80m_datasets.py
ls configs/datasets.coding80m.yaml
```

---

## 5. Install agent/preference addon

Extract:

```bash
unzip -o /path/to/gopi80m-agent-preference-addon.zip -d /tmp/gopi80m-agent-pref
```

Copy these into the project root:

```text
scripts/merge_agent_preference_sources.py
scripts/prepare_coding80m_datasets.py
scripts/validate_coding80m_prepared.py
configs/agent-preference-sources.yaml
configs/coding80m-data-prep.yaml
```

Verify:

```bash
ls \
  scripts/merge_agent_preference_sources.py \
  scripts/prepare_coding80m_datasets.py \
  scripts/validate_coding80m_prepared.py \
  configs/agent-preference-sources.yaml \
  configs/coding80m-data-prep.yaml
```

Merge the agent/preference sources into the downloader config:

```bash
python scripts/merge_agent_preference_sources.py
```

Check they were added:

```bash
grep -nE 'nvidia_swe_zero_openhands|code_edit_dpo|security_dpo' configs/datasets.coding80m.yaml
```

---

## 6. Hugging Face login

For gated datasets, accept their terms on Hugging Face first.

Then:

```bash
hf auth login
```

Or:

```bash
export HF_TOKEN='hf_your_token_here'
```

Do **not** commit the token to Git.

Check login:

```bash
hf auth whoami
```

---

## 7. Dataset downloader dry run

```bash
python scripts/download_coding80m_datasets.py \
  --config configs/datasets.coding80m.yaml \
  --profile sample \
  --dry-run
```

---

## 8. Download a sample dataset first

```bash
python scripts/download_coding80m_datasets.py \
  --config configs/datasets.coding80m.yaml \
  --profile sample \
  --continue-on-error
```

Download the dedicated agent/preference sources in sample mode:

```bash
python scripts/download_coding80m_datasets.py \
  --config configs/datasets.coding80m.yaml \
  --profile sample \
  --source nvidia_swe_zero_openhands \
  --source code_edit_dpo \
  --source security_dpo \
  --continue-on-error
```

Inspect downloaded raw data:

```bash
find data/coding80m_production/raw -maxdepth 4 -type f | head -50
```

---

## 9. Production dataset download

After checking licenses, access rights, and disk space:

```bash
python scripts/download_coding80m_datasets.py \
  --config configs/datasets.coding80m.yaml \
  --profile production \
  --continue-on-error
```

Download/continue the dedicated agent/preference datasets:

```bash
python scripts/download_coding80m_datasets.py \
  --config configs/datasets.coding80m.yaml \
  --profile production \
  --source nvidia_swe_zero_openhands \
  --source code_edit_dpo \
  --source security_dpo \
  --continue-on-error
```

The downloader is resumable. Re-running the same command continues from its download state.

Check raw size:

```bash
du -sh data/coding80m_production/raw
find data/coding80m_production/raw -name '*.jsonl' | wc -l
```

---

## 10. Back up sample/prepared data before conversion

If `data/coding80m` contains sample or previous prepared data:

```bash
mv data/coding80m "data/coding80m-backup-$(date +%Y%m%d-%H%M%S)"
```

Create the destination again only if your preparation script requires it; normally the converter creates it.

---

## 11. Test dataset preparation on 10,000 records

```bash
python scripts/prepare_coding80m_datasets.py \
  --config configs/coding80m-data-prep.yaml \
  --limit 10000 \
  --overwrite
```

Inspect:

```bash
cat data/coding80m/PREPARATION_SUMMARY.json
head -n 2 data/coding80m/pretrain_code/train.jsonl
head -n 2 data/coding80m/sft/train.jsonl
head -n 2 data/coding80m/agent/train.jsonl
head -n 2 data/coding80m/preferences/train.jsonl
```

Validate the prepared sample:

```bash
python scripts/validate_coding80m_prepared.py --root data/coding80m
```

Do not continue to production training until validation passes.

---

## 12. Full dataset preparation

```bash
python scripts/prepare_coding80m_datasets.py \
  --config configs/coding80m-data-prep.yaml \
  --overwrite
```

Validate:

```bash
python scripts/validate_coding80m_prepared.py --root data/coding80m
```

Inspect final summary:

```bash
python -m json.tool data/coding80m/PREPARATION_SUMMARY.json
```

Check counts:

```bash
for d in pretrain_code pretrain_docs pretrain_general pretrain_fim sft agent preferences; do
  echo "===== $d ====="
  wc -l "data/coding80m/$d/train.jsonl" 2>/dev/null || true
  wc -l "data/coding80m/$d/validation.jsonl" 2>/dev/null || true
done
```

Check prepared data sizes:

```bash
du -sh data/coding80m/*
```

---

## 13. Run Coding80M preflight

```bash
python coding80m/preflight.py
```

Expected end:

```text
CODING80M PREFLIGHT: PASS
```

---

## 14. Train the 8K tokenizer

> `run_tokenizer.sh` deletes the existing `data/coding80m/tokenizer` first.
> Only run this when you intentionally want to train/retrain the tokenizer.

```bash
./coding80m/run_tokenizer.sh
```

Verify vocabulary:

```bash
python coding80m/verify_vocab.py
```

Expected:

```text
VOCAB CHECK: PASS
```

Inspect tokenizer artifacts:

```bash
ls -lh data/coding80m/tokenizer
```

---

## 15. Inspect the 80M model before training

```bash
./coding80m/inspect.sh
```

The current model should be approximately:

```text
81,808,128 parameters
12 layers
hidden size 768
12 attention heads
4 KV heads
FFN 2048
vocab 8192
max model context 2048
```

---

## 16. Optional: run relevant tests before expensive training

```bash
python -m pytest -q \
  tests/test_training.py \
  tests/test_serving.py \
  tests/test_dpo.py 2>/dev/null || true
```

If your repository uses different test filenames, run your normal test suite:

```bash
python -m pytest -q
```

---

# TRAINING

## 17. Stage 1 — fresh pretraining

Check config first:

```bash
sed -n '1,260p' configs/coding80m/pretraining.stage1.yaml
```

Start Stage 1 from scratch:

```bash
./coding80m/train_stage1.sh fresh
```

Or default auto mode:

```bash
./coding80m/train_stage1.sh
```

`auto` behavior:

```text
latest.pt exists  -> --resume latest.pt
latest.pt missing -> fresh Stage 1
```

Live log:

```bash
tail -f logs/coding80m-pretrain-stage1.log
```

Status:

```bash
./coding80m/training_status.sh
```

Generate report:

```bash
./coding80m/report.sh stage1
```

Pretty-print report:

```bash
python -m json.tool reports/coding80m/pretrain-stage1.json
```

Useful training lines:

```bash
grep -Ei 'step|loss|validation|perplexity|learning|grad|tokens|checkpoint|nonfinite' \
  logs/coding80m-pretrain-stage1.log | tail -100
```

Check checkpoints:

```bash
ls -lh checkpoints/coding80m/pretrain-stage1/
```

---

## 18. If Stage 1 is interrupted

Use **resume**, not init-from:

```bash
./coding80m/train_stage1.sh resume
```

Equivalent concept:

```bash
python scripts/train.py \
  --model-config configs/coding80m/model.yaml \
  --training-config configs/coding80m/pretraining.stage1.yaml \
  --resume checkpoints/coding80m/pretrain-stage1/latest.pt
```

Resume restores the training state of the interrupted run.

---

## 19. Back up completed Stage 1

After Stage 1 successfully finishes:

```bash
mkdir -p checkpoints/coding80m/pretrain-stage1/archive
```

```bash
cp checkpoints/coding80m/pretrain-stage1/latest.pt \
  checkpoints/coding80m/pretrain-stage1/archive/latest-stage1.pt
```

```bash
cp checkpoints/coding80m/pretrain-stage1/best.pt \
  checkpoints/coding80m/pretrain-stage1/archive/best-stage1.pt
```

---

## 20. Stage 1 continuation — new phase from Stage 1 best

Use this when Stage 1 has **finished**, but validation is still improving and you intentionally want more Stage-1 pretraining.

This is a new phase, therefore use:

```text
--init-from
```

not `--resume`.

If you already have:

```text
configs/coding80m/pretraining.stage1.continue.yaml
```

inspect it:

```bash
sed -n '1,280p' configs/coding80m/pretraining.stage1.continue.yaml
```

For the continuation profile we discussed, verify values such as:

```bash
grep -nE 'samples_per_epoch|learning_rate|warmup_ratio|min_lr_ratio|output:|best_output:|log_file:|report_json:' \
  configs/coding80m/pretraining.stage1.continue.yaml
```

Typical continuation values:

```text
learning_rate: 0.00015
warmup_ratio: 0.01
min_lr_ratio: 0.10
samples_per_epoch: 2500000
```

Use separate continuation output paths, for example:

```text
checkpoints/coding80m/pretrain-stage1-cont/latest.pt
checkpoints/coding80m/pretrain-stage1-cont/best.pt
logs/coding80m-pretrain-stage1-cont.log
reports/coding80m/pretrain-stage1-cont.json
```

Start the continuation from the original Stage-1 best checkpoint:

```bash
python scripts/train.py \
  --model-config configs/coding80m/model.yaml \
  --training-config configs/coding80m/pretraining.stage1.continue.yaml \
  --init-from checkpoints/coding80m/pretrain-stage1/best.pt
```

Watch:

```bash
tail -f logs/coding80m-pretrain-stage1-cont.log
```

Build continuation report manually:

```bash
python scripts/build_training_report.py \
  --log logs/coding80m-pretrain-stage1-cont.log \
  --output reports/coding80m/pretrain-stage1-cont.json \
  --model-config configs/coding80m/model.yaml \
  --training-config configs/coding80m/pretraining.stage1.continue.yaml \
  --latest-checkpoint checkpoints/coding80m/pretrain-stage1-cont/latest.pt \
  --best-checkpoint checkpoints/coding80m/pretrain-stage1-cont/best.pt
```

Pretty print:

```bash
python -m json.tool reports/coding80m/pretrain-stage1-cont.json
```

### If the continuation is interrupted

Now use resume on the continuation checkpoint:

```bash
python scripts/train.py \
  --model-config configs/coding80m/model.yaml \
  --training-config configs/coding80m/pretraining.stage1.continue.yaml \
  --resume checkpoints/coding80m/pretrain-stage1-cont/latest.pt
```

Important:

```text
Completed old Stage 1 -> INIT-FROM old best.pt
Interrupted continuation -> RESUME continuation latest.pt
```

---

## 21. Decide which Stage-1 checkpoint feeds Stage 2

If you used the continuation phase and it produced a better validation checkpoint, archive/copy it as the Stage-1 source used for Stage 2.

Example:

```bash
mkdir -p checkpoints/coding80m/pretrain-stage1/final
```

```bash
cp checkpoints/coding80m/pretrain-stage1-cont/best.pt \
  checkpoints/coding80m/pretrain-stage1/final/best.pt
```

Your stock `train_stage2.sh` expects:

```text
checkpoints/coding80m/pretrain-stage1/best.pt
```

If the continuation is your chosen final Stage-1 checkpoint, back up the original and replace the expected source deliberately:

```bash
cp checkpoints/coding80m/pretrain-stage1/best.pt \
  checkpoints/coding80m/pretrain-stage1/best-before-continuation.pt
```

```bash
cp checkpoints/coding80m/pretrain-stage1-cont/best.pt \
  checkpoints/coding80m/pretrain-stage1/best.pt
```

Confirm:

```bash
ls -lh checkpoints/coding80m/pretrain-stage1/best*.pt
```

---

## 22. Stage 2 pretraining

Inspect config:

```bash
sed -n '1,280p' configs/coding80m/pretraining.stage2.yaml
```

Start Stage 2:

```bash
./coding80m/train_stage2.sh auto
```

For a new Stage 2, the wrapper uses:

```text
--init-from checkpoints/coding80m/pretrain-stage1/best.pt
```

If Stage 2 is interrupted:

```bash
./coding80m/train_stage2.sh resume
```

Live log:

```bash
tail -f logs/coding80m-pretrain-stage2.log
```

Report:

```bash
./coding80m/report.sh stage2
```

Pretty-print:

```bash
python -m json.tool reports/coding80m/pretrain-stage2.json
```

Check checkpoints:

```bash
ls -lh checkpoints/coding80m/pretrain-stage2/
```

---

## 23. SFT — supervised fine-tuning

Before SFT, ensure SFT + agent data are valid and non-empty:

```bash
wc -l data/coding80m/sft/train.jsonl
wc -l data/coding80m/agent/train.jsonl
```

Validate prepared data again:

```bash
python scripts/validate_coding80m_prepared.py --root data/coding80m
```

Inspect SFT config:

```bash
sed -n '1,300p' configs/coding80m/finetuning.sft.yaml
```

Start SFT:

```bash
./coding80m/train_sft.sh auto
```

A new SFT run initializes from:

```text
checkpoints/coding80m/pretrain-stage2/best.pt
```

If SFT is interrupted:

```bash
./coding80m/train_sft.sh resume
```

Live log:

```bash
tail -f logs/coding80m-sft.log
```

Report:

```bash
./coding80m/report.sh sft
```

Pretty-print:

```bash
python -m json.tool reports/coding80m/sft.json
```

Check checkpoints:

```bash
ls -lh checkpoints/coding80m/sft/
```

---

## 24. DPO — preference optimization

Only run DPO when genuine preference pairs exist:

```bash
wc -l data/coding80m/preferences/train.jsonl
wc -l data/coding80m/preferences/validation.jsonl
```

Inspect DPO config:

```bash
sed -n '1,300p' configs/coding80m/dpo.yaml
```

Start DPO:

```bash
./coding80m/train_dpo.sh auto
```

A new DPO run starts from:

```text
checkpoints/coding80m/sft/best.pt
```

and uses the SFT checkpoint as the reference checkpoint.

If DPO is interrupted:

```bash
./coding80m/train_dpo.sh resume
```

Live log:

```bash
tail -f logs/coding80m-dpo.log
```

Check checkpoints:

```bash
ls -lh checkpoints/coding80m/dpo/
```

---

## 25. One-command pipeline

Once data and tokenizer are ready, the resumable pipeline can be run with:

```bash
./coding80m/train_all.sh
```

It performs:

```text
Stage 1
  ↓
Stage 2
  ↓
SFT
  ↓
DPO if preferences/train.jsonl exists and is non-empty
```

Each stock stage auto-resumes its own `latest.pt` when present.

> If you intentionally use the separate Stage-1 continuation phase, run that phase manually before Stage 2 as shown above.

---

# MONITORING / REPORTS

## 26. Show training status

```bash
./coding80m/training_status.sh
```

---

## 27. Tail all important logs

Stage 1:

```bash
tail -f logs/coding80m-pretrain-stage1.log
```

Stage 1 continuation:

```bash
tail -f logs/coding80m-pretrain-stage1-cont.log
```

Stage 2:

```bash
tail -f logs/coding80m-pretrain-stage2.log
```

SFT:

```bash
tail -f logs/coding80m-sft.log
```

DPO:

```bash
tail -f logs/coding80m-dpo.log
```

---

## 28. Search important training metrics

```bash
grep -RniE 'loss|validation|perplexity|grad_norm|clipped_gradient_norm|learning_rate|tokens_processed|tokens_per_second|nonfinite|checkpoint' \
  logs/coding80m*.log | tail -200
```

---

## 29. Serve reports in a browser

From project root:

```bash
.venv/bin/python -m http.server 8088 \
  --bind 127.0.0.1 \
  --directory "$(pwd)/reports"
```

Open:

```text
http://127.0.0.1:8088/
```

Examples:

```text
http://127.0.0.1:8088/coding80m/pretrain-stage1.json
http://127.0.0.1:8088/coding80m/pretrain-stage1-cont.json
http://127.0.0.1:8088/coding80m/pretrain-stage2.json
http://127.0.0.1:8088/coding80m/sft.json
```

Use `8088` for reports so port `8000` remains available for model serving.

---

# FINAL SERVING

## 30. Verify final model artifacts

Tokenizer:

```bash
python coding80m/verify_vocab.py
```

Final SFT checkpoint:

```bash
ls -lh checkpoints/coding80m/sft/best.pt
```

Final DPO checkpoint, if DPO was used:

```bash
ls -lh checkpoints/coding80m/dpo/best.pt
```

---

## 31. Start the model API

```bash
./coding80m/serve.sh
```

The helper prefers:

```text
checkpoints/coding80m/dpo/best.pt
```

and falls back to:

```text
checkpoints/coding80m/sft/best.pt
```

Server address:

```text
http://127.0.0.1:8000
```

OpenAPI/Swagger, if enabled by the current serving app:

```text
http://127.0.0.1:8000/docs
```

---

## 32. Check model list

In another terminal:

```bash
curl -s http://127.0.0.1:8000/v1/models | python -m json.tool
```

---

## 33. Test chat completion

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gopi-coder-80m",
    "messages": [
      {
        "role": "user",
        "content": "Write a Python function that returns the two indices for Two Sum."
      }
    ],
    "temperature": 0.2,
    "max_tokens": 256
  }' | python -m json.tool
```

---

# RESUME / INIT-FROM / FRESH QUICK REFERENCE

## 34. When to use each mode

Use `fresh` when:

```text
You intentionally want random initialization / a new Stage-1 run.
```

Command:

```bash
./coding80m/train_stage1.sh fresh
```

Use `resume` when:

```text
The SAME training run stopped/crashed and latest.pt exists.
```

Examples:

```bash
./coding80m/train_stage1.sh resume
./coding80m/train_stage2.sh resume
./coding80m/train_sft.sh resume
./coding80m/train_dpo.sh resume
```

Use `--init-from` when:

```text
A previous phase FINISHED and you are starting a NEW phase from learned weights.
```

Examples:

```text
Stage 1 best → Stage 1 continuation
Stage 1 best → Stage 2
Stage 2 best → SFT
SFT best → DPO
```

---

# BACKUPS

## 35. Back up all final checkpoints

```bash
mkdir -p checkpoints/coding80m/release-backup
```

```bash
cp -av checkpoints/coding80m/pretrain-stage1/best.pt \
  checkpoints/coding80m/release-backup/stage1-best.pt
```

If continuation exists:

```bash
cp -av checkpoints/coding80m/pretrain-stage1-cont/best.pt \
  checkpoints/coding80m/release-backup/stage1-cont-best.pt
```

```bash
cp -av checkpoints/coding80m/pretrain-stage2/best.pt \
  checkpoints/coding80m/release-backup/stage2-best.pt
```

```bash
cp -av checkpoints/coding80m/sft/best.pt \
  checkpoints/coding80m/release-backup/sft-best.pt
```

If DPO exists:

```bash
cp -av checkpoints/coding80m/dpo/best.pt \
  checkpoints/coding80m/release-backup/dpo-best.pt
```

Hash release checkpoints:

```bash
sha256sum checkpoints/coding80m/release-backup/*.pt \
  | tee checkpoints/coding80m/release-backup/SHA256SUMS
```

---

# FINAL A-TO-Z SHORT VERSION

If all downloader/preparation files are already installed, the overall command flow is:

```bash
source .venv/bin/activate

python scripts/download_coding80m_datasets.py \
  --config configs/datasets.coding80m.yaml \
  --profile production \
  --continue-on-error

python scripts/prepare_coding80m_datasets.py \
  --config configs/coding80m-data-prep.yaml \
  --overwrite

python scripts/validate_coding80m_prepared.py --root data/coding80m

python coding80m/preflight.py

./coding80m/run_tokenizer.sh
python coding80m/verify_vocab.py
./coding80m/inspect.sh

./coding80m/train_stage1.sh fresh

# Optional/recommended longer Stage-1 continuation after Stage 1 finishes:
python scripts/train.py \
  --model-config configs/coding80m/model.yaml \
  --training-config configs/coding80m/pretraining.stage1.continue.yaml \
  --init-from checkpoints/coding80m/pretrain-stage1/best.pt

# If continuation is selected as the final Stage-1 checkpoint:
cp checkpoints/coding80m/pretrain-stage1/best.pt \
   checkpoints/coding80m/pretrain-stage1/best-before-continuation.pt

cp checkpoints/coding80m/pretrain-stage1-cont/best.pt \
   checkpoints/coding80m/pretrain-stage1/best.pt

./coding80m/train_stage2.sh auto
./coding80m/train_sft.sh auto
./coding80m/train_dpo.sh auto

./coding80m/training_status.sh
./coding80m/serve.sh
```

---

# Important stop/check points

Do not move automatically to the next phase only because a command finished. At minimum, check:

```text
Stage 1:
- validation loss trend
- perplexity trend
- nonfinite_updates = 0
- checkpoint exists

Stage 1 continuation:
- validation still improves or remains useful
- no instability after the new LR schedule

Stage 2:
- 2048-context validation remains stable
- no regression/collapse

SFT:
- held-out loss
- real prompt generations
- instruction-following behavior
- tool/agent formatting

DPO:
- chosen/rejected preference metrics
- no response collapse
- code/eval regression checks

Final:
- fixed coding benchmark suite
- syntax/compile tests
- agent repository tasks
- serving smoke tests
```

Do not treat a low training loss alone as proof that the final coding agent is good.
