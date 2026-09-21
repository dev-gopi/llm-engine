from release.artifacts import DatasetCard, ModelCard, build_reproducibility_manifest
from release.pipeline import EvaluationReleasePipeline
from evaluation.release_gate import GateRule
from training.promotion import CheckpointCandidate, MetricRule


def test_release_artifacts_and_reproducibility(tmp_path):
    config=tmp_path/"config.yaml"; config.write_text("x: 1\n")
    manifest=build_reproducibility_manifest(root=tmp_path,config_paths=[config])
    assert manifest.config_hashes
    assert ModelCard("G","1","decoder-only","use",(),(),{}, {}, {}).to_markdown().startswith("# G")
    assert DatasetCard("D","1",(),"license",(),"none",(),(),{}).to_markdown().startswith("# Dataset Card")


def test_release_pipeline_blocks_missing_p0_evidence(tmp_path):
    pipeline=EvaluationReleasePipeline(
        promotion_rules=[MetricRule("quality", protected=True)],
        regression_rules=[GateRule("quality")],
    )
    result=pipeline.run([CheckpointCandidate("ckpt",{"quality":1.0})],baseline_metrics={"quality":1.0},candidate_metrics={"quality":1.0},conformance={"api_conformance":True,"tool_calling":True,"structured_outputs":False},safety_gate=lambda:True,quality_gate=lambda:True,performance_gate=lambda:True,output_dir=tmp_path)
    assert not result.passed
    assert "structured_output_conformance:failed" in result.failures
    assert (tmp_path/"release_candidate_report.json").exists()
