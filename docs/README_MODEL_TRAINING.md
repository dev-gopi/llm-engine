# Train your MiniGPT step by step

This guide gives you a repeatable route for improving **your existing model**:
check the environment → verify the data → record a baseline → compare two short
pretraining runs → select a checkpoint → instruction-tune → test responses →
scale only the changes that helped.

There is no recipe that makes a model perfect. The objective is measurable
improvement on tasks you care about, with reproducible training and honest tests.
An 81M-parameter model needs realistic, focused targets; fluent text alone is not
proof of factual knowledge or reasoning ability.

All commands below run from the repository root. This guide describes commands
to execute; writing this guide does not start a training job.

## 1. Understand what you already have

| Component | Current setup | What to do |
|---|---|---|
| Model | MiniGPT, about 81.3M parameters, 16 layers, hidden size 512 | Keep fixed during the initial experiment |
| Attention | 8 query heads, 2 KV heads, rotary positions | Keep checkpoint-compatible settings |
| Context | 512 tokens | Prompt and generated answer must fit together |
| Base tokenizer | `data/tokenizer`, 40,000 tokens | Use throughout this guide |
| Separate extended tokenizer | `data/tokenizer-finetuning`, 42,000 tokens | Not used in this experiment |
| Existing pretrained checkpoint | `checkpoints/pretraining/latest.pt` | Preserve it |
| Frozen experiment reference | `checkpoints/pretraining-pilot/base.pt` | Initialize both comparison runs from this exact file |
| Prepared pretraining pilot | `data/cleaned/pretraining-pilot` | Ready for a small experiment |
| Prepared instruction pilot | `data/cleaned/instruction-pilot` | Ready for supervised fine-tuning |

The previous review checked the step-50,000 checkpoint, ran four diagnostic CPU
updates, and recorded 467 passing tests. Full pilot training has not yet been
completed. The diagnostic checkpoint under `checkpoints/improvement-smoke/` is
not a recommended model to use or a replacement for the frozen reference.

Read [the implementation review](PRETRAINING_IMPROVEMENTS.md) for details of fixes
to padding masks, lossless packing, document boundaries, code indentation,
held-out exclusion, and matching training/inference text normalization.

**Checkpoint:** confirm you are improving the existing model, rather than
accidentally restarting from random weights.

## 2. Decide what “better” means before training

Write down the tasks the model should handle. For this project, start with:

1. Continue a short story without changing characters or making illogical claims.
2. Answer straightforward factual questions accurately.
3. Follow an exact format, such as returning one word.
4. Solve simple arithmetic and short word problems.
5. Produce a small, correct Python function with valid indentation.
6. Answer in the languages you intend to support.

Choose acceptance targets **before looking at a new model's outputs**. For
example, use a manually checked set of questions and record correct/incorrect,
format compliance, and coherent/incoherent. The targets depend on your use case;
there is no universal perplexity value that means a model is ready.

The 10 prompts in `configs/evaluation.responses.jsonl` are development probes.
They are useful for spotting regressions but too small to measure overall
accuracy. Prepare a separate final set of unseen questions. Do not add their
answers to the training set when the model fails them.

**Checkpoint:** you have a written evaluation checklist and separate development
and final evaluation examples.

## 3. Check Python, dependencies, GPU, and disk space

```bash
cd /home/user/Downloads/llm-engine-boilerplate/llm-engine
export PYTHONPATH=src
.venv/bin/python --version
.venv/bin/python -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable")'
nvidia-smi
df -h .
```

CUDA was unavailable in the review session. The GPU training profiles refuse to
start when PyTorch cannot access CUDA. First make sure both `nvidia-smi` and the
PyTorch check work in the environment where you will train. Installing a Python
package alone does not repair an unavailable NVIDIA driver.

If the virtual environment is missing, create it and install this project:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

For CUDA, the installed PyTorch build must also match a supported GPU/driver
combination. Verify it with the command above after installation.

Allow disk space for the source datasets, generated clean/packed data, and several
independent checkpoints. A training checkpoint can include optimizer and EMA
state and be substantially larger than the model weights alone. Avoid starting
multiple training jobs on the same small GPU.

**Checkpoint:** CUDA is available for the GPU route and you have enough free disk
space for both experiment arms and the later instruction stage.

## 4. Run the implementation tests

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q
```

Resolve failures relevant to the model, loader, tokenizer, loss, checkpointing,
and generation before investing in a long run. Passing software tests means the
implementation passed those checks; it does not mean the model answers correctly.

Record the code revision and local changes used in your run:

```bash
git rev-parse HEAD
git status --short
```

Keep the exact model config, training config, tokenizer, data summary, and logs
with each experiment. If you are still editing code, finish the edit and checks
before launching the comparison.

## 5. Preserve your tokenizer and reference checkpoint

```bash
ls -lh checkpoints/pretraining-pilot/base.pt
ls -lh data/tokenizer/tokenizer.json
.venv/bin/python - <<'PY'
from tokenizer.encoder import Tokenizer
from utils.config import load_yaml

tokenizer = Tokenizer.load('data/tokenizer')
model = load_yaml('configs/model.gpu.yaml')
print('Vocabulary:', tokenizer.vocab_size)
print('Fingerprint:', tokenizer.fingerprint)
print('Context:', model['max_position'])
assert tokenizer.vocab_size == model['vocab_size'] == 40000
PY
```

Do not retrain or replace the tokenizer under an existing checkpoint. Token IDs
are part of what its weights learned. The packed data also stores tokenizer
fingerprints and will reject incompatible tokenizers.

The frozen reference was created during the review and contains the original
model weights. If it is missing, establish a new frozen reference from a chosen
checkpoint, record its identity, and use that same reference for both arms.
Do not let one arm initialize from a `latest.pt` that changes while another run
is training.

## 6. Inspect the prepared data before regenerating anything

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

for name in ('pretraining-pilot', 'instruction-pilot'):
    path = Path('data/cleaned') / name / 'summary.json'
    report = json.loads(path.read_text())
    print('\nDATASET:', name)
    for domain, details in report['domains'].items():
        if name == 'instruction-pilot':
            print(domain, 'train:', details['train'], 'validation:', details['validation'])
        else:
            print(domain, {split: details[split]['accepted_records'] for split in ('train', 'validation')})
PY
```

The prepared pretraining pilot contains 2,011 accepted source documents packed
into 3,876 training sequences. It covers TinyStories, WikiText-103, FineWeb-Edu,
and raw code. The instruction pilot contains 512 training and 64 validation
examples per domain across six domains.

Inspect actual text, not only counts. Check for broken sentences, boilerplate,
incorrect answers, repetitive examples, malformed code, and unsuitable language
balance. The filters are heuristics and do not verify every factual claim.
The pretraining preparation keeps existing source review states; its warnings
are not evidence that every source has been reviewed.

### Pretraining example

```json
{"text":"Plants use sunlight to produce sugars through photosynthesis.","source":"educational_text"}
```

The packer converts cleaned documents to bounded token sequences. It fills chunks
without throwing away every document's text after token 512. EOS marks actual
document ends. Packed `token_ids` are authoritative; `text` is a preview and may
not decode cleanly when a byte-level chunk ends inside a Unicode character.

### Instruction-tuning example

```json
{"messages":[{"role":"system","content":"You are Gopi, a helpful assistant. Answer clearly and briefly."},{"role":"user","content":"Write a Python function that adds two numbers."},{"role":"assistant","content":"def add(a, b):\n    return a + b"}]}
```

The preparation script adds the configured system formatting used in this project.
The loader trains on assistant targets while masking user/system targets.
Do not run the plain-document packer over these conversations: use the SFT
preparer to retain the instruction/answer structure and complete answers.

**Checkpoint:** samples look useful, conversations fit context, and train,
validation, and test are separated before packing.

## 7. Prepare more data when the pilot is too small

You can skip this step for the existing pilot. For a larger preparation, use a
**new output directory**, then create new configs pointing to it.

```bash
.venv/bin/python scripts/prepare_pretraining_experiment.py \
  --output data/cleaned/pretraining-expanded \
  --tokenizer data/tokenizer \
  --train-records 10000 \
  --validation-records 256 \
  --sequence-length 512 \
  --seed 42
```

These are candidate document limits per domain, not guaranteed accepted counts.
Selection samples across each entire source file. Filtering may reject records.
Set both record limits to `0` to process complete source splits; this processes
several GB and is substantially more expensive than the pilot.

The preparer excludes complete original validation/test documents against the
training candidates and shares a deduplication filter across training domains.
It uses document-level exact and SimHash matching, not exhaustive semantic or
substring contamination detection. It cannot undo exposure in old weights or an
already-trained tokenizer. Its fingerprint storage is bounded; inspect the
implementation limits before assuming exhaustive deduplication at larger scale.

Prepare a larger instruction set independently:

```bash
.venv/bin/python scripts/prepare_recovery_sft.py \
  --output data/cleaned/instruction-expanded \
  --tokenizer data/tokenizer \
  --max-length 512 \
  --train-limit 5000 \
  --validation-limit 256
```

The tokenizer-aware option rejects conversations too long for the context rather
than silently cutting off the answer. Some domains may have fewer usable examples
than requested. Preserve code indentation and inspect multilingual examples.

When changing data directories, update `train_files`, `validation_files`, and
`validation_domains` together. Keep the domain keys and their weights consistent.
Create an evaluation YAML from the same validation-domain mapping.

## 8. Understand the two pretraining profiles

| Setting | Control | Improved candidate |
|---|---|---|
| Config | `configs/pretraining.control.gpu.yaml` | `configs/pretraining.improved.gpu.yaml` |
| Initial weights | Same frozen reference | Same frozen reference |
| Training mixture | 10% TinyStories, 90% WikiText | 20% TinyStories, 35% WikiText, 40% educational text, 5% code |
| Validation | Same four domains | Same four domains |
| Sequence length | 512 | 512 |
| Batch size per GPU | 1 | 1 |
| Gradient accumulation | 32 | 32 |
| Learning rate | 0.00001 | 0.00001 |
| Epochs | 1 | 1 |
| Sampled sequences | 4,096 | 4,096 |
| Precision | FP16 | FP16 |

On one GPU:

```text
effective batch = batch size × accumulation × GPU count
                = 1 × 32 × 1 = 32 sequences

planned optimizer updates = 4096 / 32 = 128

maximum supervised tokens = 4096 × (512 - 1) = 2,093,056
```

Some packed rows are shorter, so actual tokens differ slightly between arms.
This is a matched **sequence** budget, not an exact token-budget comparison.
Weighted sampling is with replacement: increasing `samples_per_epoch` does not
create new unique documents. Do not repeat the tiny pilot for many epochs and
call that a broad training corpus.

The candidate mixture and learning rate are starting experiment choices, not
optimal values established by the review. Because both arms use corrected
packing, this experiment isolates the mixture change, not the separate effect
of fixing packing.

Optional rough resource estimate:

```bash
.venv/bin/python scripts/plan_training.py \
  --model-config configs/model.gpu.yaml \
  --training-config configs/pretraining.improved.gpu.yaml \
  --training-tokens 2093056 \
  --gpus 1 \
  --gpu-memory-gib 4
```

Use your actual GPU memory value. This is an estimate, not a guarantee that the
runtime fits. Supply explicit tokens: JSONL v2 contains token IDs and preview
text, so estimating tokens from file bytes can be misleading.

## 9. Record baseline loss and actual responses

```bash
mkdir -p reports/training-plan
.venv/bin/python scripts/evaluate_domains.py \
  --domains configs/evaluation.improved.yaml \
  --model-config configs/model.gpu.yaml \
  --training-config configs/pretraining.improved.gpu.yaml \
  --tokenizer data/tokenizer \
  --checkpoint checkpoints/pretraining-pilot/base.pt \
  > reports/training-plan/base-domains.json
```

```bash
.venv/bin/python scripts/evaluate_responses.py \
  --checkpoint checkpoints/pretraining-pilot/base.pt \
  --tokenizer data/tokenizer \
  --weights model \
  --output reports/training-plan/base-responses.json
```

The response runner also creates a Markdown file with all prompts and outputs.
It refuses to overwrite an existing response report; use a new name for another
run. The shell-redirection domain command does overwrite its output file, so use
new filenames if preserving earlier results.

For a quick execution check only, the domain evaluator supports `--max-batches 2`.
Do not report that as a full evaluation: it takes the first loader batches rather
than a representative random sample. Final comparisons should use the same full
prepared validation splits.

**Checkpoint:** baseline metrics and baseline answers are saved before training.

## 10. Train the control first

```bash
.venv/bin/python scripts/train.py \
  --model-config configs/model.gpu.yaml \
  --training-config configs/pretraining.control.gpu.yaml \
  --tokenizer data/tokenizer \
  --init-from checkpoints/pretraining-pilot/base.pt \
  --output checkpoints/pretraining-control/latest.pt \
  --best-output checkpoints/pretraining-control/best.pt \
  --log-file logs/pretraining-control.log \
  --report-json reports/pretraining-control.json
```

`--init-from` loads weights into a new training stage with a new optimizer and
schedule. Omitting both `--init-from` and `--resume` creates a randomly initialized
model, which is not the intended route for your existing checkpoint.

Use separate output directories for separate experiments. Training commands can
overwrite checkpoint output paths; never point this experiment at your original
`checkpoints/pretraining/latest.pt`.

## 11. Train the improved candidate from the same reference

Run this after the control finishes on a single small GPU:

```bash
.venv/bin/python scripts/train.py \
  --model-config configs/model.gpu.yaml \
  --training-config configs/pretraining.improved.gpu.yaml \
  --tokenizer data/tokenizer \
  --init-from checkpoints/pretraining-pilot/base.pt \
  --output checkpoints/pretraining-improved/latest.pt \
  --best-output checkpoints/pretraining-improved/best.pt \
  --log-file logs/pretraining-improved.log \
  --report-json reports/pretraining-improved.json
```

Do not initialize this arm from the control's result; that would give the two
models different training histories and invalidate the intended comparison.

## 12. Monitor the run and understand the numbers

In another terminal:

```bash
tail -f logs/pretraining-improved.log
```

| Metric | What it tells you | What to investigate |
|---|---|---|
| Training loss | Fit to sampled training examples | Non-finite values or sustained unexpected increase |
| Validation cross-entropy | Next-token prediction on held-out data | Worsening while training loss falls |
| Perplexity | Exponentiated cross-entropy; lower is better on the same evaluation | Comparing different tokenizers or changed datasets is misleading |
| Domain losses | Which capabilities are improving or regressing | A combined score can hide a weak domain |
| Gradient norm | Magnitude of gradients before clipping | Persistent spikes; context and data changes matter |
| Non-finite updates | Numerical failures or discarded updates | Repeated failures need investigation |
| Learning rate | Current point in the schedule | Accidentally restarting or resuming the wrong schedule |
| Tokens per second | Measured throughput | Falling throughput, memory pressure, other GPU processes |

Loss can include the configured z-loss penalty; use cross-entropy when
interpreting perplexity. Do not treat perplexity as percentage answer accuracy.

New educational and code examples can initially have higher loss than the old
mixture. Judge the trend on fixed validation data rather than expecting the first
batch to match the previous training loss.

The pretraining pilots evaluate/checkpoint every 100 optimizer updates and at
epoch completion. There are only 128 planned updates in each arm. In the current
instruction pilot, `evaluate_every` and `checkpoint_every` are 1,000, so its
128-update epoch relies on epoch-end evaluation/saving. If you want intermediate
instruction checkpoints, edit a copy of that profile to use a smaller interval
before starting the run. The guide does not silently change your configs.

Each command writes a separate report JSON. The existing HTML dashboard defaults
to the standard report filename; custom experiment reports are still available
as JSON. Use the per-run logs for the monitoring commands in this guide.

## 13. Resume an interrupted run correctly

Use the interrupted run's latest checkpoint and the same data/config/output paths:

```bash
.venv/bin/python scripts/train.py \
  --model-config configs/model.gpu.yaml \
  --training-config configs/pretraining.improved.gpu.yaml \
  --tokenizer data/tokenizer \
  --resume checkpoints/pretraining-improved/latest.pt \
  --output checkpoints/pretraining-improved/latest.pt \
  --best-output checkpoints/pretraining-improved/best.pt \
  --log-file logs/pretraining-improved.log \
  --report-json reports/pretraining-improved.json
```

| Situation | Correct option |
|---|---|
| Continue the same interrupted run | `--resume` |
| Start a new mixture, schedule, or instruction stage | `--init-from` |
| Start from random weights | Neither option |

Never pass `--resume` and `--init-from` together. Keep data files and sampler
settings stable when resuming. Changing them underneath a saved sampler makes
its saved position unreliable. The `--epochs` value is the total epoch target,
not “add this many more epochs after the checkpoint.”

## 14. Compare the results before choosing a winner

Run the domain evaluation from step 9 for each of:

```text
checkpoints/pretraining-control/best.pt
checkpoints/pretraining-improved/best.pt
```

Change the output filenames to `control-domains.json` and
`improved-domains.json`. Keep `configs/evaluation.improved.yaml` and the same
training/loss config in both evaluation commands.

Generate responses for each checkpoint:

```bash
.venv/bin/python scripts/evaluate_responses.py \
  --checkpoint checkpoints/pretraining-control/best.pt \
  --weights model \
  --output reports/training-plan/control-responses.json

.venv/bin/python scripts/evaluate_responses.py \
  --checkpoint checkpoints/pretraining-improved/best.pt \
  --weights model \
  --output reports/training-plan/improved-responses.json
```

Review these side by side with the baseline:

| Criterion | Baseline | Control | Improved |
|---|---|---|---|
| TinyStories cross-entropy | Fill in | Fill in | Fill in |
| WikiText cross-entropy | Fill in | Fill in | Fill in |
| Educational-text cross-entropy | Fill in | Fill in | Fill in |
| Code cross-entropy | Fill in | Fill in | Fill in |
| Factual answers correct | Fill in | Fill in | Fill in |
| Exact instructions followed | Fill in | Fill in | Fill in |
| Arithmetic answers correct | Fill in | Fill in | Fill in |
| Story coherence | Fill in | Fill in | Fill in |
| Correct small code examples | Fill in | Fill in | Fill in |

The domain evaluator loads EMA weights when available. The response runner's
`--weights model` uses raw model weights. They are separate evaluations; do not
assume one measures exactly the weights used by the other. For an EMA response
comparison, repeat with `--weights ema` and distinct report filenames. The report
records whether EMA was actually available. Raw weights are used by `--init-from`
for the next stage, so review those responses before selecting that stage's input.

If improvements are small, repeat with another fixed seed and a larger evaluation
set. Prefer a candidate that helps the target tasks without unacceptable domain
regressions. Do not automatically select the newest checkpoint or the lowest
training loss.

## 15. Instruction-tune the selected pretrained model

Pretraining learns text patterns and knowledge; SFT supplies examples of how to
answer requests. SFT does not guarantee that missing knowledge or reasoning
ability will be repaired.

The command below assumes the improved candidate won. If the control or frozen
baseline was better, replace only `--init-from` with the selected checkpoint.

```bash
.venv/bin/python scripts/train.py \
  --model-config configs/model.gpu.yaml \
  --training-config configs/finetuning.instruction-pilot.gpu.yaml \
  --tokenizer data/tokenizer \
  --init-from checkpoints/pretraining-improved/best.pt \
  --output checkpoints/instruction-pilot/latest.pt \
  --best-output checkpoints/instruction-pilot/best.pt \
  --log-file logs/instruction-pilot.log \
  --report-json reports/instruction-pilot.json
```

The current SFT pilot uses learning rate `8e-6`, batch size 1, accumulation 32,
4,096 sampled conversations, one epoch, FP16, and a 512-token context. Only
assistant targets contribute to its supervised loss, so its target-token count
can be much lower than pretraining with the same number of sequences.

Keep inference's system prompt and chat serialization consistent with SFT. Keep
using `data/tokenizer` for this checkpoint. Do not use the older serving defaults
as proof that they point at this newly trained model.

## 16. Evaluate the instruction-tuned model

First create an evaluation mapping from this exact SFT profile:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import yaml

config = yaml.safe_load(Path('configs/finetuning.instruction-pilot.gpu.yaml').read_text())
Path('reports/training-plan/instruction-domains.yaml').write_text(yaml.safe_dump({
    'domains': config['validation_domains'],
    'weights': config['validation_weights'],
}, sort_keys=False))
PY
```

```bash
.venv/bin/python scripts/evaluate_domains.py \
  --domains reports/training-plan/instruction-domains.yaml \
  --training-config configs/finetuning.instruction-pilot.gpu.yaml \
  --tokenizer data/tokenizer \
  --checkpoint checkpoints/instruction-pilot/best.pt \
  > reports/training-plan/instruction-loss.json

.venv/bin/python scripts/evaluate_responses.py \
  --checkpoint checkpoints/instruction-pilot/best.pt \
  --tokenizer data/tokenizer \
  --weights model \
  --output reports/training-plan/instruction-responses.json
```

Inspect actual answers and repeat the pretraining validation on the SFT checkpoint
if you want to measure forgetting. SFT loss and pretraining loss are different
objectives/data distributions; comparing their numbers directly does not measure
improvement.

The default response probes do not cover Bengali/Hindi despite those domains
being in the SFT mix. Add a separate reviewed multilingual prompt JSONL and pass
`--prompts PATH` when evaluating those capabilities.

Use a fresh set of factual, reasoning, formatting, and coding questions for the
final decision. Check generated code as untrusted text before executing it in an
isolated test environment. A function that looks plausible is not necessarily
correct.

## 17. Try a final interactive response

```bash
.venv/bin/python scripts/generate.py \
  'Write a short story about a girl helping a lost puppy.' \
  --model-config configs/model.gpu.yaml \
  --tokenizer data/tokenizer \
  --checkpoint checkpoints/instruction-pilot/best.pt \
  --max-tokens 96 \
  --temperature 0.2 \
  --seed 42
```

For a base model text continuation, use `--raw` and a continuation prompt:

```bash
.venv/bin/python scripts/generate.py \
  'Once upon a time, a girl found a lost puppy and decided to' \
  --checkpoint checkpoints/pretraining-improved/best.pt \
  --tokenizer data/tokenizer \
  --raw --max-tokens 96 --seed 42
```

The generation CLI uses EMA when available. For a precisely controlled raw/EMA
comparison, use the response evaluator. Keep generation settings fixed when
comparing checkpoints; changing temperature can change apparent quality without
changing the model.

## 18. Scale the successful experiment, not every setting at once

After a useful pilot result:

1. Prepare more unique, clean source documents and inspect their audit.
2. Copy the winning config to a new experiment name and update all dataset paths.
3. Keep the tokenizer and architecture unchanged initially.
4. Increase the training budget gradually and keep evaluation frequent enough to
   detect regressions within the run.
5. Retain the fixed evaluation set, model outputs, and a separate final test set.
6. Measure actual throughput, token totals, memory, and held-out improvement.
7. Change learning rate, mixture, context, or architecture in separate experiments
   when possible, so you can identify which change helped.

Do not enlarge the context only in the training YAML. It must fit the model's
position settings and GPU memory, and packed data must be rebuilt for its selected
context. Do not resize the architecture while expecting ordinary checkpoint
loading to preserve all behavior.

If the winning arm only marginally improves a small pilot, treat that as an
uncertain result. More epochs over the same few documents can memorize the pilot.

## 19. Troubleshooting

| Symptom | First checks | Next action |
|---|---|---|
| CUDA unavailable | `nvidia-smi`, PyTorch CUDA check, execution environment | Repair GPU access before the GPU route |
| Out of GPU memory | Other GPU processes, batch size, context, EMA/optimizer overhead | Use one job; keep batch 1 and checkpointing; test a copied smaller profile if necessary |
| Tokenizer fingerprint mismatch | Check checkpoint, tokenizer directory, and packing metadata | Restore the matching tokenizer or repack from original documents |
| Packed record exceeds context | Compare packer's sequence length with training config | Repack at the intended context instead of silently truncating |
| NaN/Inf or frequent discarded updates | Data, loss scale, learning rate, precision support | Reproduce with a small batch; inspect numerical failures before continuing |
| Train loss improves, validation worsens | Split quality, repeats, overfitting | Select the earlier checkpoint; improve data or reduce training duration |
| Fluent but wrong answers | Factual coverage and actual response evaluations | Improve reliable knowledge/task coverage; do not use loss as answer accuracy |
| Ignores instructions | Base vs SFT checkpoint, system prompt, role template | Train/evaluate correctly formatted SFT examples |
| Answers end mid-sentence | `finish_reason`, output token cap, remaining context | Adjust the generation budget within context before calling it a learned EOS problem |
| Code indentation disappears | Preparation version and regenerated data | Rebuild old data using the corrected whitespace-preserving pipeline |
| Domain score improves but answers worsen | EMA/raw choice, changed prompts, seed, data mix | Compare like-for-like outputs and per-domain regressions |
| Very few SFT examples survive | Quality rejection and `over_context` statistics | Inspect examples; curate shorter complete answers or plan a compatible larger-context experiment |

## 20. What to archive when you finish

Save the chosen checkpoint and tokenizer together with:

- Model and training YAMLs, plus the inference settings used in evaluation.
- Code revision and relevant local changes.
- Data preparation summaries, source identities, and tokenizer fingerprint.
- Train/validation/test definitions and the actual training token count.
- Loss results for each domain and actual prompt/response files.
- GPU/precision settings, run duration, seed, and any interruption/resume history.
- A short note stating what improved, what failed, and what remains untested.

A run is successful when it produces a reproducible improvement on your intended
tasks. Finishing an epoch or writing `best.pt` is only an intermediate milestone.
