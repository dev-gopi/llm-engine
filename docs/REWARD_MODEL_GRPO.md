# Reward model and offline GRPO

## Pairwise reward-model training

Input uses the existing preference JSONL contract:

```json
{"prompt":"Explain 2 + 2","chosen":"2 + 2 equals 4.","rejected":"2 + 2 equals 5."}
```

Start from the best SFT/aligned causal checkpoint:

```bash
python scripts/train_reward_model.py \
  --model-config configs/model.gpu.yaml \
  --training-config configs/reward_model.gpu.yaml \
  --init-from checkpoints/sft/best.pt
```

Resume the same run with `--resume checkpoints/reward-model/latest.pt`.

The model attaches a scalar reward head to the decoder backbone and scores the
final non-padding hidden state. Training minimizes a pairwise logistic ranking
loss so chosen responses receive larger rewards than rejected responses.

## Offline GRPO

Offline GRPO consumes complete prompt groups with externally computed rewards:

```json
{"prompt":"What is 2 + 2?","completions":["4","5","It is four."],"rewards":[1.0,-1.0,0.9]}
```

Every record must contain at least two completions, all groups in one dataset
must use the same group size, and equal-reward groups are rejected because they
contain no group-relative signal.

```bash
python scripts/train_grpo.py \
  --model-config configs/model.gpu.yaml \
  --training-config configs/grpo.gpu.yaml \
  --reference-checkpoint checkpoints/sft/best.pt \
  --init-from checkpoints/sft/best.pt
```

The initial policy checkpoint is frozen as the old-policy baseline by default.
A different baseline can be supplied with `--old-policy-checkpoint`. The path is
stored in checkpoint metadata so a resumed run uses the same baseline unless an
explicit override is provided.

This is intentionally an offline/pre-scored implementation. It does not claim
to provide online rollout workers, tool/verifier execution, reward-model calls
during rollout generation, or iterative old-policy refresh.
