"""Run the release-candidate gate: conformance -> quality/performance/safety -> regression -> promotion."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.release_gate import GateRule
from release.artifacts import write_json
from release.pipeline import EvaluationReleasePipeline
from training.promotion import CheckpointCandidate, MetricRule


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/release")
    parser.add_argument("--metrics", required=True, help="JSON metrics file produced by the model evaluation benchmark")
    args=parser.parse_args()
    output=Path(args.output); output.mkdir(parents=True, exist_ok=True)
    metrics=json.loads(Path(args.metrics).read_text())
    required_metrics={"quality","latency","safety"}
    missing=required_metrics-set(metrics)
    if missing:
        raise SystemExit("metrics file is missing required release gates: " + ", ".join(sorted(missing)))
    conformance_code=subprocess.call([sys.executable,"scripts/run_api_conformance.py"])
    conformance={
        "api_conformance": conformance_code == 0,
        "tool_calling": conformance_code == 0,
        "structured_outputs": conformance_code == 0,
    }
    candidate_metrics={"quality":float(metrics.get("quality",0.0)),"latency":float(metrics.get("latency",0.0))}
    baseline_metrics={"quality":float(metrics.get("baseline_quality",candidate_metrics["quality"])),"latency":float(metrics.get("baseline_latency",candidate_metrics["latency"]))}
    candidate=CheckpointCandidate("configured-candidate",candidate_metrics,{"release_blocked":False})
    pipeline=EvaluationReleasePipeline(
        promotion_rules=[MetricRule("quality", direction="max", protected=True), MetricRule("latency", direction="min", protected=True)],
        regression_rules=[GateRule("quality", direction="max", max_regression=float(metrics.get("max_quality_regression",0.0)))],
    )
    result=pipeline.run(
        [candidate], baseline_metrics=baseline_metrics, candidate_metrics=candidate_metrics,
        conformance=conformance, safety_gate=lambda: float(metrics.get("safety",0.0)) >= 1.0,
        quality_gate=lambda: candidate_metrics["quality"] >= float(metrics.get("min_quality",0.0)),
        performance_gate=lambda: candidate_metrics["latency"] > 0 and candidate_metrics["latency"] <= float(metrics.get("max_latency",float("inf"))),
        output_dir=output, manifest_kwargs={"root":"."},
    )
    write_json(output/"release_result.json",result.to_dict())
    print(json.dumps(result.to_dict(),indent=2,sort_keys=True))
    return 0 if result.passed else 1

if __name__ == "__main__": raise SystemExit(main())
