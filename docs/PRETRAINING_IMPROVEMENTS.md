# MiniGPT fixes and controlled improvement experiment

The checkpoint can produce prose, but the recorded response probes show factual,
reasoning, and instruction-following failures. The fixes below correct verified
implementation/data problems. They do not establish that the existing weights
have become a capable assistant.

## Verified fixes

- MiniGPT now converts its binary attention mask to boolean. Previously a float
  0/1 mask reached attention as an additive bias, allowing padding to influence
  predictions. The normal training collator already uses boolean masks, so this
  bug is not evidence that the previous training run was broken.
- Packed JSONL v2 stores authoritative token IDs and verifies tokenizer identity.
  Decoded text is only a preview. This avoids decode/re-encode changes and Unicode
  corruption at byte-token boundaries. Eager and lazy loaders reject a packed
  sequence that would otherwise be silently truncated by a shorter context.
- The packer fills sequential chunks and appends EOS only at document ends,
  rather than teaching EOS at artificial splits. The smallest supported context
  (2 tokens) now works. Rebuild old packs from original documents; old text-only
  packs remain readable but cannot retroactively recover discarded bytes.
- Text preprocessing and instruction-data cleaning preserve code indentation.
  Chat training and inference now share Unicode/newline normalization; all
  prepared instruction examples were checked for identical encoded sequences.
- Instruction preparation excludes *all* validation/test prompts from training,
  including quality-rejected and unsampled validation rows. Selected training
  prompts are also deduplicated across domains. Optional tokenizer-aware filtering
  keeps complete conversations that fit the context instead of truncated answers.
- The original pretraining evaluation weights now match the active 10%/90% mix.

The architecture and tokenizer vocabulary are preserved for checkpoint compatibility.

## Prepared local pilot

`data/cleaned/pretraining-pilot/summary.json` records candidate counts, rejection
reasons, accepted text hashes, source file metadata, and packing metadata. Selection
uses deterministic reservoir sampling over each entire train/validation file.
The pilot considers 512 training documents and 64 validation documents per domain.
It reserves the complete original validation and test splits against training.
The shared filter performs exact and SimHash near-duplicate exclusion across
training sources. This is not exhaustive substring or semantic contamination
screening, and it cannot undo exposure in an existing checkpoint or tokenizer.

The candidate improved mixture is 20% TinyStories, 35% WikiText-103, 40% FineWeb-Edu,
and 5% raw code. These are experiment settings, not proven optimal proportions.
Existing source manifests are copied with their original review states.

The control and improved profiles use the same packed pipeline, architecture,
optimizer settings, seed, and four-domain validation. The control retains the
original 10% TinyStories / 90% WikiText training mixture. Thus this comparison
isolates the *mixture* change; it does not separately measure the effect of packing.

Each profile samples 4,096 sequences for one pilot epoch: at most 2,093,056 target
tokens (4,096 × 511). Final partial rows make the actual token totals slightly
smaller. This is a matched sequence budget, not an exact token-budget experiment.
Sampling is with replacement; a small pilot may repeat documents. For a longer
run, prepare more unique source documents first.

Instruction data is in `data/cleaned/instruction-pilot/`: 512 training examples
and 64 validation examples per domain (chat, English, Bengali, Hindi, math, coding).
Its summary includes the tokenizer fingerprint and 512-token context limit.
Filtering is heuristic and does not certify the correctness of every answer.

## Reproduce preparation

The pretraining preparer requires an empty output directory to avoid accidentally
replacing a previous comparison. Choose a fresh path for a new preparation, and
update the train/validation paths in the configs accordingly.

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_pretraining_experiment.py \
  --output data/cleaned/pretraining-pilot-new \
  --train-records 512 --validation-records 64
```

For all source documents, set both record limits to `0`. That processes several
GB of local source data and takes substantially longer than this pilot.

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_recovery_sft.py \
  --output data/cleaned/instruction-pilot-new \
  --tokenizer data/tokenizer --max-length 512 \
  --train-limit 512 --validation-limit 64
```

## Run the controlled training experiment

A model-only frozen reference is saved at `checkpoints/pretraining-pilot/base.pt`.
Both arms must initialize from that exact reference using `--init-from`, which
starts a new optimizer/schedule. Do not resume an old optimizer into this stage.
The short smoke checkpoint is only a diagnostic and should not be promoted.

These profiles are intended for a working CUDA training environment. CUDA was
unavailable during this review; CPU smoke updates were used for verification.
The full pilot training arms have not been run.

```bash
PYTHONPATH=src .venv/bin/python scripts/train.py \
  --training-config configs/pretraining.control.gpu.yaml \
  --tokenizer data/tokenizer --init-from checkpoints/pretraining-pilot/base.pt \
  --output checkpoints/pretraining-control/latest.pt \
  --best-output checkpoints/pretraining-control/best.pt \
  --log-file logs/pretraining-control.log \
  --report-json reports/pretraining-control.json
```

```bash
PYTHONPATH=src .venv/bin/python scripts/train.py \
  --training-config configs/pretraining.improved.gpu.yaml \
  --tokenizer data/tokenizer --init-from checkpoints/pretraining-pilot/base.pt \
  --output checkpoints/pretraining-improved/latest.pt \
  --best-output checkpoints/pretraining-improved/best.pt \
  --log-file logs/pretraining-improved.log \
  --report-json reports/pretraining-improved.json
```

Evaluate the reference and both resulting checkpoints on the same validation
files and fixed prompts. Use distinct output names for each response evaluation.
The response runner uses the same configured system prompt formatter as generation
and the prepared SFT examples, with no retrieval or tools. Its default is raw
model weights; the domain evaluator uses EMA if present, so note the weight choice
when comparing the two evaluation types.

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_domains.py \
  --domains configs/evaluation.improved.yaml \
  --training-config configs/pretraining.improved.gpu.yaml \
  --checkpoint checkpoints/pretraining-pilot/base.pt
```

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_responses.py \
  --checkpoint checkpoints/pretraining-pilot/base.pt \
  --output reports/pilot-base-responses.json
```

Keep test splits and additional prompts for final assessment. The checked-in
response probes are a small development set with explicit review criteria, not
an overall accuracy benchmark. Do not train on these probe outputs.

## Instruction-tune the selected checkpoint

After comparing the two pretraining arms, substitute the selected checkpoint in
this command. This stage preserves the 40,000-token pretraining tokenizer and
uses assistant-only training targets with complete in-context answers.

```bash
PYTHONPATH=src .venv/bin/python scripts/train.py \
  --training-config configs/finetuning.instruction-pilot.gpu.yaml \
  --tokenizer data/tokenizer \
  --init-from checkpoints/pretraining-improved/best.pt \
  --output checkpoints/instruction-pilot/latest.pt \
  --best-output checkpoints/instruction-pilot/best.pt \
  --log-file logs/instruction-pilot.log \
  --report-json reports/instruction-pilot.json
```

Repeat response evaluation using `--tokenizer data/tokenizer`. Do not pair this
checkpoint with the existing 42,000-token fine-tuning tokenizer unless explicitly
performing a verified vocabulary-extension stage. Existing serving defaults point
to another fine-tuning checkpoint; this pilot has not been deployed.

## Verification completed

- Full repository suite: **467 passed** (26.35 seconds).
- Real step-50,000 checkpoint: 81,314,304 parameters, all finite; causal attention
  and cached/full inference parity passed.
- Four CPU optimizer updates on educational text, raw code, math instructions,
  and coding instructions completed with zero non-finite updates. Gradient norms
  were high on the new domains (54–96 before clipping); these are diagnostic
  samples, not a learning curve or evidence of quality improvement.
- Every generated packed row loaded within the 512-token context. Every prepared
  SFT example has supervised target tokens, fits context, and encodes identically
  through the training and inference formatting paths.
- Detailed evidence: `reports/minigpt_improvements.json`. Actual baseline responses:
  `reports/improvement-baseline-responses.md`.

The broader pretraining and instruction training stages are prepared, but have
not been completed. The smoke checkpoint is isolated from the original weights.
