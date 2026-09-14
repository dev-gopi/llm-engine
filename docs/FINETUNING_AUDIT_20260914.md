# Completed fine-tuning audit — 2026-09-14

## First load the completed run

The user confirmed v2 finished on the external drive. The supplied chat command
loads **v1**, `checkpoints/finetuning/best.pt`, with the 42K tokenizer. It does
not load the v2 artifacts named in `finetuning.gpu.v2.yaml` and `tokenizer.v2.yaml`.

After mounting that drive, run:

```bash
.venv/bin/python scripts/chat.py --inference-config configs/inference.v2.yaml
```

Chat now prints the checkpoint path, training step, selected weights, tokenizer
path, vocabulary size, and fingerprint prefix. `--weights model` selects raw
weights; the default is EMA when available. Compare both before choosing one.
Do not retrain or extend the tokenizer of an already completed checkpoint.

**Scope:** the external v2 directory is unavailable in this workspace, and
PyTorch reports CUDA unavailable. All checkpoint-quality numbers below refer
to local v1, not v2. No completed checkpoint was overwritten or retrained.
The optional pilot has not been run on the real model.

## Measurements and performance labels

Labels assess the stated area, not overall intelligence. GOOD means the reviewed
behavior has supporting checks; UNKNOWN means the necessary measurement is
unavailable. No area is rated EXCELLENT without evidence.

| Area | Label | Evidence and limitation |
|---|---|---|
| Architecture | GOOD | Consistent 16-layer, width-512 decoder; 8 query/2 KV heads, RoPE, RMSNorm, SwiGLU, tied embeddings. Model/cache/causality tests pass. No parameter shapes changed. |
| V1 capability | POOR | Chat-aligned development probes: 5/18. Stricter six-domain retention probes: 4/30. Local official GSM8K test subset: 0/50 under the recorded final-answer protocol. |
| Completed v2 capability | UNKNOWN | Checkpoint, tokenizer, logs, and external datasets cannot be read here. |
| Dataset mixture | GOOD | Fixed four unmatched names. Intended weights now resolve exactly; missing/unused/nonfinite weights fail before indexing data. Actual learned v2 sampling history remains unverified. |
| Gradient accumulation | GOOD | Unequal masked microbatches and partial windows now match the full-batch SGD update within test tolerances. Empty-target windows leave weights, scheduler and EMA unchanged. |
| Numerical stability | GOOD | Local SFT checkpoint records 384,540,320 supervised tokens and zero nonfinite updates. New trainer tests pass; CUDA precision behavior was not measured. |
| Learning rate / training budget | FAIR | V2's 5e-6 LR and effective batch 64 are conservative, not proven optimal. No LR sweep or completed-v2 loss history is available. |
| Tokenizer implementation | GOOD | 180 multilingual/code records encode to identical 27,189 token IDs and identical fingerprint before/after matcher caching. Warm median encoding time: 1.128 s → 0.676 s in this CPU sample. V2 tokenizer quality remains unmeasured. |
| Broad local data | POOR | In 10,746 inspected records: 1,753 overlength, 334 with zero retained supervised targets, 56 normalized full-record duplicates. Unsupported/incomplete tool chats were also found. |
| Recovery local data | FAIR | Original full prompt scan found three Bengali split overlaps. The separate prepared copy retains 66,320 records with zero exact train/validation prompt overlap; semantic correctness remains unverified. |
| Evaluation / retention | FAIR | Reproducible provenance, strict expression/numeric scoring, per-case regression rejection and initial checkpoint preservation are implemented. Small development suites do not cover every capability. |
| Inference / checkpoint loading | GOOD | All 18 chat-aligned baseline answers remain identical. CPU deserialization avoids moving unused optimizer/EMA payloads onto the GPU. Actual GPU peak savings were not measured. |
| Context handling | FAIR | Trained maximum remains 1,024 tokens. Overlength SFT requires filtering or semantically valid splitting, not a speculative RoPE/context increase. |
| GPU utilization / throughput | UNKNOWN | NVIDIA driver and CUDA are unavailable. No GPU speed or VRAM claim is made. |
| Post-training approach | FAIR | Clean SFT plus existing replay is implemented as an optional guarded pilot. Learned improvement requires evaluating and training the actual v2 artifacts. |

The local model has **82,338,304 parameters** at 42K vocabulary. A full 44K
extension would have 83,362,304; that is an estimate, not an inspection of v2.
FP32 v1 parameters occupy about 314 MiB. A single BF16 full-context KV cache is
estimated at 8 MiB; optimizer state, EMA, activations and logits are additional.

The older pretraining checkpoint's scheduler is at step 96,000 despite a saved
46,875-step schedule, and has reached its LR floor. This may reflect an intentional
continuation or changed run settings; the artifact alone cannot establish which.
The local SFT schedule ends at its configured 31,250 steps. New post-training
stages must initialize weights with fresh optimizer/scheduler state.

## Baseline and forgetting evidence

The legacy evaluator injected a safety paragraph despite
`embed_safety_instruction: false`. Its 3/18 result became 5/18 when evaluated
with the actual chat template. **This is a protocol correction, not learning.**
Both raw and EMA local-v1 weights scored 5/18. These prompts are development
checks, not standardized benchmark accuracy. “Thank you for your help!” occurs
in sampled tool training records, so it is explicitly not an independent test.

For language retention, the pretraining checkpoint (step 96,000) and SFT
checkpoint (step 31,250) were evaluated on the same 24 unique, complete validation
texts per source. Each used its own exact tokenizer and EMA weights. Texts had
to fit both tokenizers within 512 tokens; EOS was scored. Bits per UTF-8 byte
permits comparison of identical text across the changed tokenization; token
perplexity alone would not.

| Source | Pretraining bits/byte | SFT bits/byte | Relative change |
|---|---:|---:|---:|
| TinyStories | 0.6506 | 0.8670 | +33.26% |
| WikiText-103 | 1.0783 | 1.5565 | +44.35% |
| FineWeb-Edu | 2.0116 | 1.4716 | −26.84% |
| Code pretraining | 4.1146 | 1.4428 | −64.93% |

Higher is worse. This supports a retention tradeoff in the local SFT run, not
a diagnosis of the inaccessible v2 run or a corpus-wide estimate. The samples
are deterministic bounded prefixes after length/duplicate filtering. Full
training contamination and semantic near-duplicates were not exhaustively checked.

The additional GSM8K baseline uses 50 hash-selected locally stored records
marked `official_split: test`, excluding normalized prompts in the local GSM8K
training file. It uses greedy decoding, 192 output tokens and an explicit
`####` numeric final answer. Three responses reached the length limit. The
result is **0/50** for this subset/protocol, not the full official benchmark.
Overlap with every other training source remains unchecked. Frozen cases are
in `reports/audit-20260914-gsm8k-cases.jsonl` (SHA256
`d467a50eefa91856f719a1636823b8a85706775fa961bf02432cb7a199a98dc7`).

## Modification record

1. **V2 weights:** changed `recovery_chat/english/math/coding` keys to the actual
   directory names. Previously the loader defaulted each to 1.0, making their
   combined sampling probability 86.39% instead of 37%, and Aya Bengali/Hindi
   each 3.02% instead of 14%. This calculation assumes the checked-in code/config
   and nonempty sources were used. Tests now cover every configured mixture.
2. **Token-weighted accumulation:** normalized the sum of supervised-token losses
   across the accumulation window, with distributed gradient-average correction.
   Before the fix, the unequal-length fixture differed by up to 0.024743 in an
   embedding weight; full and partial-window tests now pass at atol 2e-7/rtol 2e-5.
   No CUDA or multi-rank performance measurement was available.
3. **Empty windows:** prevented optimizer decay, scheduler advance and EMA updates
   when truncation/masking supplies no supervised targets. The pre-fix regression
   test failed; the corrected test passes.
4. **Tokenizer speed:** cached the extension matcher and direct token lookup;
   preserved the tokenizer artifact, vocabulary IDs, fingerprints and segmentation.
5. **Inference:** added `inference.v2.yaml`, explicit weight selection in chat,
   startup identity output, exact tokenizer matching for inference, and CPU-first
   loading. Append-only initialization remains supported for a new training stage.
6. **Prompt parity / EMA reporting:** evaluation, generation, serving and training
   generation checks respect the configured prompt flag. Fixed a response-report
   bug that always reported EMA unused. Optional EMA loss evaluation restores raw
   training weights afterward; the pilot disables EMA to avoid short-run lag and
   its parameter copy. This pilot choice has not been validated on v2.
7. **Evaluation:** recorded case hash, prompt, decoding settings, tokenizer,
   checkpoint identity/step, weight choice, timing and finish reasons. Added
   operator-preserving code scoring and explicit numeric-answer scoring. Comparisons
   reject changed protocols, changed coverage, and any lost passing probe.
8. **Checkpoint retention:** optionally saves the initial evaluated model before
   updates, then promotes only strict overall improvements that retain every
   previously passing case. Required evaluation failures stop training. Training
   resume checkpoints and inference-only generation checkpoints remain separate.
9. **Data:** the auditor now measures the actual role template, BOS/EOS and masks,
   and reports per-file truncation, empty supervision and split/benchmark overlap.
   Fixed Unicode prompt identity in recovery preparation. Added a separate SFT
   preparation stage that preserves chat structure/indentation, reserves validation,
   available sibling test sets and supplied probes, uses disk-backed exact dedup,
   and rejects malformed/incomplete/overlength chats. It does not rewrite answers
   or certify semantic correctness. Original corpora are preserved.
10. **Optional pilot:** requires prepared data, a matching tokenizer, an explicit
    initialization checkpoint and CUDA; uses batch 1 × accumulation 64, separate
    output paths, a 25K-example budget and checks every 100 updates. These are
    bounded pilot settings, not a claim of optimal hyperparameters.

## Why SFT and replay, not another RL algorithm

The observed faults include wrong artifact selection, incorrect sampling,
inconsistent evaluation and weak basic task competence. Correct these before
judging the completed model. Existing sources already cover the requested domains;
there is no measured justification to download another corpus while v2 is unseen.

For a demonstrated remaining gap, the implemented approach is complete, cleaned
supervised examples mixed with existing recovery/replay data and retention gates.
Defer DPO/RL until v2 has a stable baseline and suitable verified preferences or
rewards. DPO optimizes preferences; its existence does not establish it will repair
this model's arithmetic. EWC, an extra KL teacher, LoRA, and architecture growth
were not added without comparative evidence.

This decision is an engineering inference from the local findings. Relevant
primary references: the [gradient-accumulation correction](https://huggingface.co/blog/gradient_accumulation),
the [continual-fine-tuning forgetting study](https://arxiv.org/abs/2308.08747), and
the [DPO paper](https://arxiv.org/abs/2305.18290).

## Evaluate v2 before considering the pilot

Run from the repository root after the drive is mounted:

```bash
V2_ROOT=/media/user/c55a3051-add0-438c-95e3-28a4c4f7d44b/llm-engine-v2
.venv/bin/python scripts/evaluate_benchmarks.py \
  --cases configs/evaluation.retention.jsonl \
  --inference-config configs/inference.v2.yaml \
  --tokenizer "$V2_ROOT/tokenizer-finetuning-v2" \
  --checkpoint "$V2_ROOT/checkpoints/finetuning-v2/best.pt" \
  --weights ema --max-tokens 48 --output reports/v2-baseline-ema.json
```

Repeat with `--weights model --output reports/v2-baseline-model.json`. Use the same
cases/settings for candidates; `--baseline reports/v2-baseline-ema.json` writes
the comparison and exits 2 for regressions. Reuse the frozen GSM8K case file with
192 output tokens for that separate baseline. Also inspect actual explanations,
stories and generated functions: scalar probes do not establish those qualities.

If v2 still needs correction, prepare its existing data into a new directory:

```bash
.venv/bin/python scripts/prepare_sft_stage.py \
  --training-config configs/finetuning.v2.retention-pilot.gpu.yaml \
  --output data/processed/v2-retention-clean \
  --cases configs/evaluation.domains.jsonl \
  --cases configs/evaluation.retention.jsonl \
  --cases data/processed/gsm8k/test.jsonl
```

Inspect the generated `audit.json` and retained examples in every domain. Raw replay
documents retain their causal objective and may be context-truncated by the loader;
chat records must fit completely. No raw-text flattening of SFT is performed.
Incorrect but well-formed answers still require review. If the pilot is warranted:

```bash
.venv/bin/python scripts/train.py \
  --model-config configs/model.gpu.yaml \
  --training-config data/processed/v2-retention-clean/training.yaml \
  --init-from "$V2_ROOT/checkpoints/finetuning-v2/best.pt"
```

Evaluate `checkpoints/v2-retention-pilot/best-generation.pt` against the original
v2 baseline, including raw-text retention. It is inference-only; resume interrupted
training from `latest.pt`, never from `best-generation.pt`. Do not overwrite or
automatically replace the completed v2 checkpoint. Promotion still needs broader
domain and qualitative evidence beyond this small selection suite.

## Evidence files and remaining work

Local JSON artifacts under `reports/audit-20260914-*` record inventory, before/after
tokenizer timings and token-ID hashes, legacy/chat baselines, the 30-case suite,
GSM8K subset, paired raw-text losses, data audits, full recovery prompt overlaps,
resolved v2 mixture weights and the final generation comparison. Generated reports
are ignored by Git; this document and the reusable tools/tests are versionable.

Code verification: the original suite passed 605 tests before modifications;
the expanded suite passed **629 tests in 24.16 seconds**. Python compilation and
`git diff HEAD --check` also passed. A subprocess integration test verifies that
the initial retention checkpoint is saved before the first optimizer update and
survives a tied score afterward. These tests establish execution/correctness,
not learned model-quality improvement.

The final same-protocol generation comparison passed with **zero lost cases**,
zero change in each domain's score, and **18/18 identical answers** against the
pre-fix chat-aligned baseline. Model and EMA tensors in both inspected best
checkpoints were all finite (131 model state tensors, 130 EMA tensors each).

Actual data preparation also completed on the local recovery corpus:
`data/processed/recovery_sft_audited_20260914_complete/`. Across its 12 files,
66,330 records were read and 66,320 retained. Five held-out prompt overlaps
(three Bengali split overlaps plus two reserved chat probes) and five duplicate
prompts were excluded. No overlength, malformed-chat or empty-supervision rows
were found in that full prepared recovery run. All output file hashes match the
audit, and the disk-backed full prompt intersection is zero. The generated config
requires explicit checkpoint initialization. This local copy uses the v1 tokenizer;
prepare the v2 mixture with its exact v2 tokenizer before using the optional pilot.
The earlier interrupted preparation directory without the `_complete` suffix has
no training configuration and is not a usable prepared stage.

The completed v2 baseline, its actual training history, full external-data audit,
GPU memory/speed measurements, and any learned improvement/regression comparison
remain pending access to that drive and GPU. Do not infer v2 quality from v1's
scores or confuse passing code tests with improved model capability.

## Follow-up: DPO correctness review

Three new regression tests failed before the fixes and passed afterward:

- Preference input validation accepted null fields as literal `None` text and
  silently truncated complete preference answers, including pairs with no
  response supervision. It now rejects non-string fields, overlength pairs,
  and token-identical alternatives; valid complete pairs are retained.
- Validation averaged batch means, overweighting the final partial batch.
  The regression fixture scored 23.0 with batch size 1 but 20.75 with batch
  size 2. Metrics now weight each preference pair equally; training epoch
  summaries use the same weighting.
- With gradient clipping disabled, non-finite gradients could update and
  corrupt policy weights. Gradient finiteness is now checked independently
  of clipping; the failing update raises before optimizer/scheduler progress.

The focused DPO/scaling suite passed 27 tests. These changes affect future DPO
runs only; no preference training was launched or existing checkpoint modified.
Rejecting overlength pairs can reduce usable data, so review retained pair counts
before a future DPO run. This review does not certify GPU behavior or model quality.

After these fixes, the full CPU test suite passed **632 tests in 22.63 seconds**.
