# V2 model training guide

This guide is for a new user who wants to train, evaluate, align, export, and
run the v2 model. Run every command from the repository root. Do not run two
training stages at the same time on one GPU.

The complete order is:

```text
Environment and data checks
        -> tokenizer (fresh models only)
        -> pretraining
        -> optional continued pretraining
        -> supervised fine-tuning (SFT)
        -> preference dataset
        -> DPO
        -> final evaluation
        -> export and generation
```

## 1. Create the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --editable '.[dev]'
```

Run the test suite before starting a long training job:

```bash
.venv/bin/python -m pytest
```

Check CUDA, BF16, FP16, SDPA, and other accelerator capabilities with the
repository's hardware-check script:

```bash
.venv/bin/python scripts/capabilities.py
```

The v2 GPU fine-tuning profile requires a CUDA GPU with BF16 support. Use the
matching CPU profiles on a machine without CUDA.

## 2. Collect and prepare datasets

Skip individual commands in this section when their processed `train.jsonl`,
`validation.jsonl`, and `test.jsonl` files already exist. Downloads require an
internet connection and can be large. Read each source dataset card and license
before downloading or training. The commands do not grant permission to use a
dataset.

The general Hugging Face collector downloads Parquet shards, normalizes common
text/instruction/chat schemas, and creates deterministic disjoint splits:

```bash
.venv/bin/python scripts/prepare_hf_dataset.py \
  --dataset OWNER/DATASET \
  --full \
  --output-dir data/processed/LOCAL_NAME
```

Use the following commands for the active v2 datasets.

### Pretraining data

Collect TinyStories:

```bash
.venv/bin/python scripts/prepare_hf_dataset.py \
  --dataset roneneldan/TinyStories \
  --full \
  --raw-dir data/raw/tinystories \
  --output-dir data/processed/tinystories
```

WikiText requires its official train, validation, and test Parquet shards. The
temporary processed directory created by the downloader is not used for
training; `prepare_wikitext.py` rebuilds article-level records from the raw
shards:

```bash
.venv/bin/python scripts/prepare_hf_dataset.py \
  --dataset Salesforce/wikitext \
  --config wikitext-103-raw-v1 \
  --split train \
  --full \
  --raw-dir data/raw/wikitext-103-raw-v1 \
  --output-dir data/staging/wikitext

.venv/bin/python scripts/prepare_hf_dataset.py \
  --dataset Salesforce/wikitext \
  --config wikitext-103-raw-v1 \
  --split validation \
  --full \
  --raw-dir data/raw/wikitext-103-raw-v1 \
  --output-dir data/staging/wikitext

.venv/bin/python scripts/prepare_hf_dataset.py \
  --dataset Salesforce/wikitext \
  --config wikitext-103-raw-v1 \
  --split test \
  --full \
  --raw-dir data/raw/wikitext-103-raw-v1 \
  --output-dir data/staging/wikitext

.venv/bin/python scripts/prepare_wikitext.py \
  --raw-dir data/raw/wikitext-103-raw-v1 \
  --output-dir data/processed/wikitext_103
```

### Main SFT data

Collect UltraChat's raw official SFT splits, then create the bounded working
subset:

```bash
.venv/bin/python scripts/prepare_hf_dataset.py \
  --dataset HuggingFaceH4/ultrachat_200k \
  --split train_sft \
  --full \
  --raw-dir data/raw/ultrachat_200k \
  --output-dir data/staging/ultrachat

.venv/bin/python scripts/prepare_hf_dataset.py \
  --dataset HuggingFaceH4/ultrachat_200k \
  --split test_sft \
  --full \
  --raw-dir data/raw/ultrachat_200k \
  --output-dir data/staging/ultrachat

.venv/bin/python scripts/prepare_ultrachat.py \
  --raw-dir data/raw/ultrachat_200k \
  --output-dir data/processed/ultrachat_200k
```

Collect HelpSteer, OpenOrca, and GSM8K. HelpSteer rating fields are retained so
the same processed records can later produce DPO pairs:

```bash
.venv/bin/python scripts/prepare_hf_dataset.py \
  --dataset nvidia/HelpSteer \
  --full \
  --output-dir data/processed/helpsteer

.venv/bin/python scripts/prepare_hf_dataset.py \
  --dataset Open-Orca/OpenOrca \
  --full \
  --output-dir data/processed/openorca

.venv/bin/python scripts/prepare_hf_dataset.py \
  --dataset openai/gsm8k \
  --config main \
  --full \
  --output-dir data/processed/gsm8k

.venv/bin/python scripts/prepare_core_chat.py \
  --output-dir data/processed/core_chat
```

### Capability and multilingual SFT data

```bash
.venv/bin/python scripts/prepare_hf_dataset.py --full \
  --dataset iamtarun/python_code_instructions_18k_alpaca \
  --output-dir data/processed/code_instructions

.venv/bin/python scripts/prepare_hf_dataset.py --full \
  --dataset databricks/databricks-dolly-15k \
  --output-dir data/processed/general_qa

.venv/bin/python scripts/prepare_hf_dataset.py --full \
  --dataset fwnlp/self-instruct-safety-alignment \
  --output-dir data/processed/safety_alignment

.venv/bin/python scripts/prepare_hf_dataset.py --full \
  --dataset HuggingFaceH4/no_robots \
  --output-dir data/processed/writing_editing

.venv/bin/python scripts/prepare_hf_dataset.py --full \
  --dataset rishiraj/bengalichat \
  --output-dir data/processed/multilingual_bn_hi

.venv/bin/python scripts/prepare_hf_dataset.py --full \
  --dataset rishiraj/hindichat \
  --output-dir data/processed/multilingual_hi

.venv/bin/python scripts/prepare_hf_dataset.py --full \
  --dataset narrative-io/narrative-function-calling-v1 \
  --output-dir data/processed/tool_calling

.venv/bin/python scripts/prepare_hf_dataset.py --full \
  --dataset kamruzzaman-asif/bangla-instruction-dataset \
  --config QApair \
  --output-dir data/processed/bangla_qa

.venv/bin/python scripts/prepare_hf_dataset.py --full \
  --dataset kamruzzaman-asif/bangla-instruction-dataset \
  --config RQA \
  --output-dir data/processed/bangla_reading_qa

.venv/bin/python scripts/prepare_hf_dataset.py --full \
  --dataset kaifahmad/indian-history-hindi-QA-3.4k \
  --output-dir data/processed/hindi_history_qa

.venv/bin/python scripts/prepare_hf_dataset.py --full \
  --dataset DSMJ910/hinglish-instruct-10k \
  --output-dir data/processed/hinglish_chat

.venv/bin/python scripts/prepare_hf_dataset.py --full \
  --dataset flwrlabs/code-alpaca-20k \
  --output-dir data/processed/code_alpaca
```

`emoji_chat` is a small project-authored dataset, not a third-party download.
Keep the supplied `data/processed/emoji_chat` directory. A replacement must use
the same JSONL chat structure (`id`, `source`, `bot_name`, and alternating
`messages`) and disjoint train, validation, and test files.

Do not assume that a successful download is ready for production use. Inspect
sample rows, remove invalid/duplicate/private content, verify train/validation
separation, and create or review `dataset-manifest.yaml` before training.

## 3. Check the processed datasets

The commands below assume that the processed datasets already exist under
`data/processed`. Check the two pretraining datasets first:

```bash
wc -l \
  data/processed/tinystories/train.jsonl \
  data/processed/tinystories/validation.jsonl \
  data/processed/wikitext_103/train.jsonl \
  data/processed/wikitext_103/validation.jsonl
```

Check every dataset referenced by the SFT configuration with the repository's
dataset-audit command:

```bash
.venv/bin/python scripts/audit_datasets.py \
  --training-config configs/finetuning.v2.gpu.yaml \
  --stage sft
```

Audit dataset governance before training:

```bash
.venv/bin/python scripts/audit_datasets.py \
  --training-config configs/pretraining.v2.gpu.yaml \
  --stage pretraining
```

The audit checks file presence as well as manifests, provenance, licensing,
privacy review, and allowed training stages. A `missing_file` finding means the
dataset must be prepared before training. Governance findings such as
`missing_manifest` or `privacy_unreviewed` need review, but do not by themselves
mean that the JSONL file is absent or malformed.

## 4. Prepare the tokenizer

### Existing pretrained model

If `checkpoints/v2-pretraining/latest.pt` or `best.pt` already exists, keep the
tokenizer that trained it:

```text
data/tokenizer-v2
```

Do **not** retrain or replace this tokenizer. A rebuilt vocabulary can assign
different meanings to existing token IDs and make the checkpoint incompatible.

Inspect the existing tokenizer:

```bash
.venv/bin/python scripts/tokenize.py inspect \
  "Hello বাংলা हिन्दी 👋" \
  --tokenizer data/tokenizer-v2 \
  --add-bos \
  --add-eos
```

### Completely fresh model only

Run this only when no model has been trained with `data/tokenizer-v2`, or when
intentionally beginning a new model family from scratch:

```bash
.venv/bin/python scripts/tokenize.py train \
  --config configs/tokenizer.v2.yaml
```

After a model has been pretrained, use the append-only tokenizer extension
workflow instead of rebuilding its vocabulary. See the tokenizer section in
the main README before choosing that advanced workflow.

## 5. Base pretraining

### Start a fresh pretraining run

Do not add `--resume` to a fresh run:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/train.py \
  --model-config configs/model.v2.gpu.yaml \
  --training-config configs/pretraining.v2.gpu.yaml \
  --tokenizer data/tokenizer-v2 \
  --output checkpoints/v2-pretraining/latest.pt \
  --best-output checkpoints/v2-pretraining/best.pt
```

`latest.pt` is for recovery. `best.pt` is selected using validation and is the
checkpoint passed to the next stage.

### Resume an interrupted pretraining run

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/train.py \
  --model-config configs/model.v2.gpu.yaml \
  --training-config configs/pretraining.v2.gpu.yaml \
  --tokenizer data/tokenizer-v2 \
  --resume checkpoints/v2-pretraining/latest.pt \
  --output checkpoints/v2-pretraining/latest.pt \
  --best-output checkpoints/v2-pretraining/best.pt
```

Do not change tokenizer, dataset weights, validation metric, model architecture,
or epoch plan in the middle of a resumed run. A rare skipped non-finite FP16
update is recoverable; repeated or rapidly increasing non-finite updates require
investigation.

Training progress logs include:

```text
epoch, step, current and average loss, learning rate, gradient norm
processed tokens, tokens/second, progress %, elapsed seconds, ETA seconds
best validation loss, peak GPU memory, allocated/reserved/total GPU memory
discarded non-finite update count
```

`gpu_memory_mb=A/R/T` means currently allocated, currently reserved by PyTorch,
and total device memory. ETA is estimated from measured training-step time and
does not include future validation, checkpoint I/O, pauses, or system slowdown.
When validation reaches a new minimum, a separate `new_best_validation` log
records the previous loss, new loss, step, and metric name.

## 6. Evaluate pretraining

```bash
.venv/bin/python scripts/evaluate_domains.py \
  --domains configs/evaluation.v2.pretraining.yaml \
  --model-config configs/model.v2.gpu.yaml \
  --training-config configs/pretraining.v2.gpu.yaml \
  --tokenizer data/tokenizer-v2 \
  --checkpoint checkpoints/v2-pretraining/best.pt \
  --device cuda
```

Compare TinyStories and WikiText separately. The aggregate value alone can hide
a weak domain.

Test raw text completion:

```bash
.venv/bin/python scripts/generate.py \
  "Once upon a time" \
  --model-config configs/model.v2.gpu.yaml \
  --tokenizer data/tokenizer-v2 \
  --checkpoint checkpoints/v2-pretraining/best.pt \
  --device cuda \
  --raw \
  --max-tokens 120 \
  --temperature 0.7
```

## 7. Optional continued pretraining

Skip this stage unless a separate continued-pretraining experiment is wanted.
For the existing TinyStories/WikiText data, use one epoch in
`configs/pretraining.v2.continued.gpu.yaml`. Do not overwrite the base
pretraining checkpoints.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/train.py \
  --model-config configs/model.v2.gpu.yaml \
  --training-config configs/pretraining.v2.continued.gpu.yaml \
  --tokenizer data/tokenizer-v2 \
  --init-from checkpoints/v2-pretraining/best.pt \
  --output checkpoints/v2-pretraining-continued/latest.pt \
  --best-output checkpoints/v2-pretraining-continued/best.pt
```

Use `--init-from`, not `--resume`, when starting this new stage. If this stage
is skipped, use `checkpoints/v2-pretraining/best.pt` as the SFT input. If it is
run and improves validation, use
`checkpoints/v2-pretraining-continued/best.pt` instead.

## 8. Supervised fine-tuning (SFT)

The following command assumes that optional continued pretraining was skipped:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/train.py \
  --model-config configs/model.v2.gpu.yaml \
  --training-config configs/finetuning.v2.gpu.yaml \
  --tokenizer data/tokenizer-v2 \
  --init-from checkpoints/v2-pretraining/best.pt \
  --output checkpoints/v2-finetuning/latest.pt \
  --best-output checkpoints/v2-finetuning/best.pt
```

When the optional stage was used successfully, replace the `--init-from` value
with:

```text
checkpoints/v2-pretraining-continued/best.pt
```

Resume an interrupted SFT run with its complete state:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/train.py \
  --model-config configs/model.v2.gpu.yaml \
  --training-config configs/finetuning.v2.gpu.yaml \
  --tokenizer data/tokenizer-v2 \
  --resume checkpoints/v2-finetuning/latest.pt \
  --output checkpoints/v2-finetuning/latest.pt \
  --best-output checkpoints/v2-finetuning/best.pt
```

Never provide `--resume` and `--init-from` together.

## 9. Evaluate SFT quality

Evaluate English, Bengali, Hindi, coding, GSM8K, and chat independently:

```bash
.venv/bin/python scripts/evaluate_domains.py \
  --domains configs/evaluation.v2.finetuning.yaml \
  --model-config configs/model.v2.gpu.yaml \
  --training-config configs/finetuning.v2.gpu.yaml \
  --tokenizer data/tokenizer-v2 \
  --checkpoint checkpoints/v2-finetuning/best.pt \
  --device cuda
```

Run deterministic capability cases:

```bash
.venv/bin/python scripts/evaluate_benchmarks.py \
  --cases configs/evaluation.v2.domains.jsonl \
  --model-config configs/model.v2.gpu.yaml \
  --tokenizer data/tokenizer-v2 \
  --checkpoint checkpoints/v2-finetuning/best.pt \
  --device cuda
```

## 10. Build the DPO preference dataset

DPO records require `prompt`, `chosen`, and `rejected`. Build deterministic
pairs from processed HelpSteer data:

```bash
.venv/bin/python scripts/prepare_helpsteer_preferences.py \
  --input-dir data/processed/helpsteer \
  --output-dir data/processed/preferences
```

Confirm that both files contain data:

```bash
wc -l \
  data/processed/preferences/train.jsonl \
  data/processed/preferences/validation.jsonl
```

## 11. DPO preference training

Run DPO only after SFT. The frozen reference and initial policy both start from
the validation-selected SFT checkpoint:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/train_dpo.py \
  --model-config configs/model.v2.gpu.yaml \
  --training-config configs/dpo.v2.gpu.yaml \
  --tokenizer data/tokenizer-v2 \
  --reference-checkpoint checkpoints/v2-finetuning/best.pt \
  --init-from checkpoints/v2-finetuning/best.pt \
  --output checkpoints/v2-dpo/latest.pt \
  --best-output checkpoints/v2-dpo/best.pt \
  --device cuda
```

Resume interrupted DPO training:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/train_dpo.py \
  --model-config configs/model.v2.gpu.yaml \
  --training-config configs/dpo.v2.gpu.yaml \
  --tokenizer data/tokenizer-v2 \
  --reference-checkpoint checkpoints/v2-finetuning/best.pt \
  --resume checkpoints/v2-dpo/latest.pt \
  --output checkpoints/v2-dpo/latest.pt \
  --best-output checkpoints/v2-dpo/best.pt \
  --device cuda
```

The DPO CLI is single-device. Do not launch it with `torchrun` or multiple
processes.

## 12. Test the final checkpoint

English:

```bash
.venv/bin/python scripts/generate.py \
  "Explain artificial intelligence in simple language." \
  --model-config configs/model.v2.gpu.yaml \
  --tokenizer data/tokenizer-v2 \
  --checkpoint checkpoints/v2-dpo/best.pt \
  --device cuda \
  --max-tokens 150 \
  --temperature 0.7
```

Bengali:

```bash
.venv/bin/python scripts/generate.py \
  "বাংলায় কম্পিউটার কী তা সহজভাবে ব্যাখ্যা করো।" \
  --model-config configs/model.v2.gpu.yaml \
  --tokenizer data/tokenizer-v2 \
  --checkpoint checkpoints/v2-dpo/best.pt \
  --device cuda \
  --max-tokens 150 \
  --temperature 0.7
```

Use the SFT `best.pt` instead when DPO was intentionally skipped.

## 13. Export the final model

Export the DPO checkpoint as a SafeTensors bundle:

```bash
.venv/bin/python scripts/export.py \
  --model-config configs/model.v2.gpu.yaml \
  --tokenizer data/tokenizer-v2 \
  --checkpoint checkpoints/v2-dpo/best.pt \
  --format safetensors \
  --output exports/v2-final/gopi-v2.safetensors
```

This also copies `model.yaml` and the tokenizer into `exports/v2-final`.

Generate directly from the exported bundle:

```bash
.venv/bin/python scripts/generate_exported.py \
  "Hello, explain what a computer is." \
  --model exports/v2-final/gopi-v2.safetensors \
  --model-config exports/v2-final/model.yaml \
  --tokenizer exports/v2-final/tokenizer \
  --device cuda \
  --max-tokens 150 \
  --temperature 0.7 \
  --top-k 40 \
  --top-p 0.9
```

## Checkpoint rules

- Use `latest.pt` only to resume the same interrupted stage.
- Use `best.pt` to initialize the next stage, evaluate, generate, or export.
- Use `--resume` to restore model, optimizer, scheduler, scaler, and progress.
- Use `--init-from` to begin a new stage from model weights.
- Never use `--resume` and `--init-from` together.
- Never rebuild the tokenizer used by an existing checkpoint.
- Keep each stage in a different checkpoint directory.
- Stop training cleanly with `Ctrl+C`, then resume from that stage's
  `latest.pt`.

## Recommended final artifact

If all stages are completed, deploy:

```text
checkpoints/v2-dpo/best.pt
```

or its exported bundle:

```text
exports/v2-final/gopi-v2.safetensors
exports/v2-final/model.yaml
exports/v2-final/tokenizer/
```
