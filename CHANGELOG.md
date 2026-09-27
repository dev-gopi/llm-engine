
## Batch 22 — semantic cache production layer
- Added optional Redis semantic-cache backend with distributed single-flight primitives.
- Added tenant quotas, privacy/no-store policy, sensitive-data exclusion, dynamic-request freshness fingerprints, negative caching, warming, selective purge, and route/model/task thresholds.
- Added cache metrics for token/latency savings and similarity distributions plus explainable selection metadata.
- Added offline semantic-cache false-positive/answer-drift evaluation tooling.
- Preserved the existing local SQLite semantic cache and public behavior.
