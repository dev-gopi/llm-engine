# Reward scoring, calibration, and online GRPO

## Reward-model calibration

Calibrate a trained reward model on held-out chosen/rejected pairs:

```bash
python scripts/calibrate_reward_model.py \
  --model-config configs/model.gpu.yaml \
  --tokenizer data/tokenizer-finetuning \
  --checkpoint checkpoints/reward-model/best.pt \
  --input data/processed/preferences/validation.jsonl \
  --output checkpoints/reward-model/calibration.json \
  --max-sequence-length 512
```

The calibration artifact stores reward mean/std normalization, a bounded
normalization clip and a pairwise temperature. The command also reports
pairwise accuracy, ties, pairwise NLL and Brier score. Calibration does not
change raw reward ordering.

Score arbitrary prompt/completion records:

```bash
python scripts/score_rewards.py \
  --model-config configs/model.gpu.yaml \
  --tokenizer data/tokenizer-finetuning \
  --checkpoint checkpoints/reward-model/best.pt \
  --calibration checkpoints/reward-model/calibration.json \
  --normalized \
  --input data/candidates.jsonl \
  --output data/candidates.scored.jsonl
```

Input records use `prompt` and `completion`; output preserves the record and
adds `reward` plus `reward_normalized`.

## Generate online rollout groups without training

```bash
python scripts/generate_grpo_rollouts.py \
  --model-config configs/model.gpu.yaml \
  --tokenizer data/tokenizer-finetuning \
  --policy-checkpoint checkpoints/sft/best.pt \
  --reward-checkpoint checkpoints/reward-model/best.pt \
  --reward-calibration checkpoints/reward-model/calibration.json \
  --normalize-reward-model \
  --prompt-file data/processed/grpo/prompts.jsonl \
  --output data/processed/grpo/live.jsonl \
  --group-size 4
```

Prompt files are JSONL records containing at least `prompt`. Records may also
contain `expected_answer`, `answer`, or `expected` when the exact-match reward
signal is enabled. Failed prompt rollouts are isolated instead of aborting the
whole generation job.

## Online GRPO training

```bash
python scripts/train_grpo_online.py \
  --model-config configs/model.gpu.yaml \
  --training-config configs/grpo.online.gpu.yaml \
  --tokenizer data/tokenizer-finetuning \
  --reference-checkpoint checkpoints/sft/best.pt \
  --reward-checkpoint checkpoints/reward-model/best.pt \
  --reward-calibration checkpoints/reward-model/calibration.json \
  --init-from checkpoints/sft/best.pt
```

Each online iteration:

1. snapshots the current policy into the frozen old-policy baseline;
2. samples a fresh completion group for each prompt;
3. scores completions with the reward model and optional exact-match signal;
4. writes rollout/failure artifacts atomically;
5. combines fresh groups with the bounded replay buffer when configured;
6. performs the configured GRPO updates;
7. saves a resumable checkpoint with iteration and reward provenance.

Resume with `--resume checkpoints/grpo-online/latest.pt`.

The current online implementation is single-process by design. Distributed or
asynchronous rollout workers, tool/code verifiers, remote reward services and
FSDP online updates are intentionally not advertised as implemented.
