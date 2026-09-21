# Contributing

Thank you for contributing to llm-engine.

## Before you start

- Read the repository's `AGENTS.md` and the relevant entry in
  `docs/PROJECT_INDEX.md`.
- Keep changes focused and preserve compatibility with existing checkpoints and
  configurations.
- Discuss substantial architectural changes in an issue before implementation.

## Development workflow

1. Create a branch from the current default branch.
2. Make the smallest safe change in the authoritative source file.
3. Add or update focused tests.
4. Run the relevant tests with the project virtual environment, for example:

   ```bash
   .venv/bin/pytest tests/test_attention.py -q
   ```

5. Update user-facing documentation when behavior changes.
6. Open a pull request using the provided template.

## Pull request expectations

Explain the problem, summarize the solution, list validation performed, and
call out checkpoint/configuration compatibility or hardware-memory effects.
Do not include credentials, private datasets, or generated model weights.

## Reporting problems

Use the bug-report template for reproducible defects and the feature-request
template for proposals. Do not report security vulnerabilities in public issues;
follow [SECURITY.md](SECURITY.md) instead.
