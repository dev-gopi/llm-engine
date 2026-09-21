# ADR-0010: Model Growth and Progressive Depth Expansion

## Status
`ACCEPTED`

## Context
Pretraining large language models (e.g. 500M or 1B+ parameters) from scratch using randomly initialized Gaussian weights requires extensive GPU compute time and risks early optimization instability.

## Decision
We implemented a **Progressive Model Growth Engine** ([`src/training/model_growth.py`](../../src/training/model_growth.py)) supporting depth expansion and append-only vocabulary expansion:
1. Copies base transformer layers directly to target layers.
2. Newly appended transformer layers are initialized as **identity residuals** by zeroing attention output projections ($W_{\text{out}} = 0$) and feed-forward down-projections ($W_{\text{ffn\_down}} = 0$).
3. The initial grown model produces mathematically identical predictions to the smaller base model, allowing warm-started continuation of training.

## Alternatives Considered
- **Random Gaussian Weight Initialization**: Slower convergence; completely discards pre-existing learned representations of the base model.
- **Direct Layer Duplication without Zero-Init**: Drastically perturbs residual stream dynamics, causing sudden loss spikes.

## Consequences
- **Positive**: Seamlessly scales model capacity (e.g., 16 layers $\rightarrow$ 32 layers) preserving base performance; saves up to 40% pretraining compute.
- **Negative**: Requires strict validation of parameter shapes and initialization contracts.

## Related Files
- [`src/training/model_growth.py`](../../src/training/model_growth.py)
- [`scripts/grow_checkpoint.py`](../../scripts/grow_checkpoint.py)
- [`tests/test_model_growth.py`](../../tests/test_model_growth.py)

