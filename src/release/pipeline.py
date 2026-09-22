"""Checkpoint -> evaluation -> regression -> promotion -> release-candidate pipeline."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.release_gate import GateRule, ReleaseGate
from training.promotion import CheckpointCandidate, CheckpointPromoter, MetricRule

from .artifacts import build_reproducibility_manifest, write_json


@dataclass(frozen=True)
class ReleaseResult:
    passed: bool
    checkpoint: str | None
    stages: Mapping[str, bool]
    failures: tuple[str, ...]
    report_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "checkpoint": self.checkpoint, "stages": dict(self.stages), "failures": list(self.failures), "report_path": self.report_path}


class EvaluationReleasePipeline:
    def __init__(self, *, promotion_rules: list[MetricRule], regression_rules: list[GateRule]) -> None:
        self.promoter = CheckpointPromoter(promotion_rules)
        self.regression_gate = ReleaseGate(regression_rules)

    @staticmethod
    def _run_stage(name: str, stage: Callable[[], Any], failures: list[str]) -> bool:
        try:
            value = stage()
            passed = bool(value.get("passed", False)) if isinstance(value, Mapping) else bool(value)
        except Exception as error:
            failures.append(f"{name}:{type(error).__name__}:{error}")
            return False
        if not passed:
            failures.append(f"{name}:failed")
        return passed

    def run(
        self,
        candidates: list[CheckpointCandidate],
        *,
        baseline_metrics: Mapping[str, float] | None = None,
        candidate_metrics: Mapping[str, float] | None = None,
        conformance: Mapping[str, bool] | None = None,
        quality_gate: Callable[[], Any] | None = None,
        performance_gate: Callable[[], Any] | None = None,
        safety_gate: Callable[[], Any] | None = None,
        output_dir: str | Path | None = None,
        manifest_kwargs: Mapping[str, Any] | None = None,
    ) -> ReleaseResult:
        failures: list[str] = []
        stages: dict[str, bool] = {}
        if conformance:
            stages["api_conformance"] = all(conformance.values())
            if not stages["api_conformance"]: failures.append("api_conformance:failed")
        else:
            stages["api_conformance"] = False
            failures.append("api_conformance:missing")
        for name, gate in (("tool_conformance", "tool_calling"), ("structured_output_conformance", "structured_outputs")):
            stages[name] = bool(conformance and conformance.get(gate, False))
            if not stages[name]: failures.append(f"{name}:failed")
        stages["safety"] = self._run_stage("safety", safety_gate, failures) if safety_gate else False
        if safety_gate is None: failures.append("safety:missing")
        stages["quality"] = self._run_stage("quality", quality_gate, failures) if quality_gate else False
        if quality_gate is None: failures.append("quality:missing")
        stages["performance"] = self._run_stage("performance", performance_gate, failures) if performance_gate else False
        if performance_gate is None: failures.append("performance:missing")
        if baseline_metrics is not None and candidate_metrics is not None:
            regression = self.regression_gate.evaluate(dict(baseline_metrics), dict(candidate_metrics))
            stages["regression"] = bool(regression["passed"])
            if not stages["regression"]: failures.extend(f"regression:{x}" for x in regression["failures"])
        else:
            stages["regression"] = False
            failures.append("regression:missing")

        decision = self.promoter.decide(candidates)
        stages["promotion"] = decision.promoted
        if not decision.promoted: failures.extend(f"promotion:{x}" for x in decision.failures)
        passed = not failures and decision.promoted
        report_path = None
        if output_dir is not None:
            output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
            report = {"passed": passed, "checkpoint": decision.selected, "stages": stages, "failures": failures, "promotion": {"selected": decision.selected, "scores": dict(decision.scores)}}
            write_json(output / "release_candidate_report.json", report)
            report_path = str(output / "release_candidate_report.json")
            if passed and manifest_kwargs is not None:
                manifest = build_reproducibility_manifest(**dict(manifest_kwargs))
                write_json(output / "reproducibility_manifest.json", manifest.to_dict())
                (output / "RELEASE_CANDIDATE").write_text(decision.selected or "", encoding="utf-8")
        return ReleaseResult(passed, decision.selected, stages, tuple(failures), report_path)
