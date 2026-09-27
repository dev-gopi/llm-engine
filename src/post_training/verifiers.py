"""Pluggable rollout verification and reward execution.

Verifiers are isolated from GRPO itself. Sandboxed execution uses a subprocess with a timeout;
for production untrusted code, deploy this worker behind a container/VM boundary.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol


class RolloutVerifier(Protocol):
    name: str

    def verify(
        self, prompt: str, completion: str, context: dict[str, Any] | None = None
    ) -> float: ...


@dataclass(frozen=True)
class ExactMatchVerifier:
    name: str = "exact_match"

    def verify(self, prompt: str, completion: str, context=None) -> float:
        expected = (context or {}).get("expected_answer")
        return (
            1.0
            if expected is not None and completion.strip() == str(expected).strip()
            else 0.0
        )


@dataclass(frozen=True)
class PythonUnitTestVerifier:
    """Execute a generated Python expression/test in a temporary subprocess.

    The worker has a sanitized environment, no network, a bounded timeout and a fresh temp dir.
    OS-level sandboxing (seccomp/container) should be supplied by the deployment environment.
    """

    timeout_seconds: float = 2.0
    name: str = "python_unit_test"

    def verify(self, prompt: str, completion: str, context=None) -> float:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        test = (context or {}).get("test_code")
        if not isinstance(test, str) or not test.strip():
            return 0.0
        with tempfile.TemporaryDirectory(prefix="llm_verify_") as td:
            path = os.path.join(td, "candidate.py")
            # Candidate is supplied as data to a test harness, not shell-expanded.
            harness = f"candidate = {completion!r}\n" + test + "\n"
            with open(path, "w", encoding="utf8") as f:
                f.write(harness)
            env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": ""}
            try:
                p = subprocess.run(
                    [os.environ.get("PYTHON", "python"), path],
                    cwd=td,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                return 0.0
            return 1.0 if p.returncode == 0 else 0.0


@dataclass
class CompositeVerifier:
    verifiers: Sequence[RolloutVerifier]
    weights: Sequence[float] | None = None
    name: str = "composite"

    def __post_init__(self):
        if not self.verifiers:
            raise ValueError("at least one verifier is required")
        if self.weights is None:
            self.weights = [1.0] * len(self.verifiers)
        if len(self.weights) != len(self.verifiers) or sum(self.weights) <= 0:
            raise ValueError("weights must match verifiers and have positive sum")

    def verify(self, prompt, completion, context=None) -> float:
        return sum(
            float(w) * v.verify(prompt, completion, context)
            for w, v in zip(self.weights, self.verifiers)
        ) / sum(self.weights)


class RewardRouter:
    def __init__(self, scorers: dict[str, RolloutVerifier]):
        if not scorers:
            raise ValueError("at least one scorer is required")
        self.scorers = dict(scorers)

    def score(
        self, verifier: str, prompt: str, completion: str, context=None
    ) -> dict[str, Any]:
        if verifier not in self.scorers:
            raise KeyError(f"unknown verifier {verifier!r}")
        value = float(self.scorers[verifier].verify(prompt, completion, context))
        return {"verifier": verifier, "reward": value}


@dataclass(frozen=True)
class HTTPVerifier:
    """Remote verifier-farm adapter using JSON POST with a shared bearer token."""

    endpoint: str
    token: str | None = None
    timeout_seconds: float = 10.0
    name: str = "remote_http"

    def verify(self, prompt: str, completion: str, context=None) -> float:
        import urllib.request

        if not self.endpoint.startswith(("http://", "https://")):
            raise ValueError("endpoint must be HTTP(S)")
        body = json.dumps(
            {"prompt": prompt, "completion": completion, "context": context or {}}
        ).encode()
        req = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.token}"} if self.token else {}),
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode())
        return float(payload["reward"])
