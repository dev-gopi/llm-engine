# ADR-0008: Progressive Context Documentation System for AI Coding Agents

## Status
`ACCEPTED`

## Context
When AI coding agents collaborate on large repositories, standard workflows frequently cause agents to scan or ingest the entire codebase into their prompt context. This wastes hundreds of thousands of context tokens, increases inference latency, and induces hallucinations.

## Decision
We designed a **Hierarchical Progressive Context Documentation System**:
1. Single root entry point: [`AGENTS.md`](../../AGENTS.md).
2. Machine-friendly lookup index: [`docs/PROJECT_INDEX.md`](../PROJECT_INDEX.md).
3. Subsystem-specific compact context packs: [`docs/context/*.context.md`](../context) (<200 lines each).
4. Authoritative source of truth pointers rather than duplicated code snippets.

## Alternatives Considered
- **Single Monolithic README.md**: Overwhelms context window; difficult to maintain and navigate.
- **Auto-generated Docstrings Only**: Lacks high-level architectural rationale, invariants, and hardware boundaries.

## Consequences
- **Positive**: Reduces onboarding token consumption by >80%; allows agents to locate and edit target files in under 3 steps; preserves cross-session project memory.
- **Negative**: Requires engineers and agents to update context packs when core contracts evolve.

## Related Files
- [`AGENTS.md`](../../AGENTS.md)
- [`docs/PROJECT_INDEX.md`](../PROJECT_INDEX.md)
- [`docs/context/`](../context)

