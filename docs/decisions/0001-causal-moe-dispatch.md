# ADR 0001: Preserve causal predictions under MoE capacity limits

- **Date**: 2026-10-05
- **Status**: Accepted

## Context

SparseMoE previously ranked routes across the flattened batch and sequence and
dropped routes exceeding an expert's capacity. Future tokens could displace an
earlier token's route. Evaluation also applied this rule, making predictions
depend on batch composition and differ between prefill and cached decoding.

Top-1 routing separately normalized its one selected logit with softmax, making
its weight a constant one and removing the prediction-loss gradient to the
router.

## Decision

Capacity controls bound each expert invocation. Process every selected route
in chunks of at most the calculated capacity instead of dropping overflow.
Apply the same dispatch semantics in training, evaluation, and the native
expert-parallel exchange path. Preserve the existing capacity calculation and
configuration keys; retain the dropped-route metric with value zero.

For top-1 routing, use the selected expert's full-router softmax probability.
For top-k with k > 1, retain softmax normalization over the selected experts.
Use the same top-1 weighting in the sharded inference adapter.

## Consequences

- With stochastic regularization disabled, token predictions are independent of
  expert congestion elsewhere in the sequence or batch.
- Top-1 routers receive prediction-loss gradients without requiring an
  auxiliary objective.
- Capacity limits bound per-invocation work, not total retained training
  activations. Processing all routes may cost more compute than dropping them.
- Existing checkpoint keys and tensor shapes remain compatible. Top-1 and
  capacity-limited MoE outputs intentionally change and require reevaluation.
- The default dense architecture and active GPU training settings are unchanged.
