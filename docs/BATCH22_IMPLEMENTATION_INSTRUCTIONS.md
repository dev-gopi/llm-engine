# Batch 22 Implementation Instructions

## Root
This batch continues from the Batch-21 patched repository. Do not reset or replace existing implementations.

## Scope
Implement the next production semantic-cache batch from the current missing-feature audit:

19. Distributed Redis/vector-index semantic-cache backend.
20. Cross-replica single-flight coordination.
21. Multi-tenant/user cache namespaces with quotas.
22. Per-request `no-store` / privacy control.
23. Automatic sensitive-data cache exclusion policy/classifier.
24. RAG/tool/web/session freshness fingerprints.
25. Negative/error caching with separate short TTL.
26. Cache warming/prepopulation.
27. Selective purge by model/tenant/time/key.
28. Token/cost/latency-saved and similarity-distribution metrics.
29. Explain/debug metadata for semantic-match selection.
30. Offline false-positive / answer-drift semantic-cache evaluation tooling.
31. Stale-while-revalidate/background refresh.
32. Per-route/model/task similarity thresholds.

## Safety / compatibility rules
- Preserve the existing `SemanticResponseCache` API and SQLite/local behavior.
- Redis is optional; importing the package must not require a running Redis server.
- Dynamic requests must remain bypassed unless a caller explicitly supplies a freshness fingerprint and an explicit cache policy.
- `no-store` must never write or read a cached response.
- Sensitive-data exclusion is conservative and configurable; false positives are preferable to caching likely secrets/PII.
- Negative/error entries must never be returned as successful generations.
- Tenant quotas must be enforced independently of the global capacity.
- Cross-replica single-flight must fail open to normal generation if Redis is unavailable.
- Background refresh must never block the original request path.
- Metrics/debug metadata must not expose prompts or secrets by default.
- Do not claim a feature is production-qualified without a real Redis integration test; CPU unit tests may establish contract correctness only.

## Required validation
1. Existing semantic-cache tests remain green.
2. New unit tests cover every item above.
3. Redis-dependent tests use a skip marker when Redis is unavailable and must never make the whole suite fail merely because Redis is absent.
4. `python -m compileall -q src scripts` passes.
5. Update `TASKS.md`, `CHANGELOG.md`, and the missing-feature audit only for features actually implemented and contract-tested.
