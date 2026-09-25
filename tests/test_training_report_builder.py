import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "build_training_report.py"
REPORT_TEMPLATE = Path(__file__).parents[1] / "reports" / "training_report.html"
SPEC = importlib.util.spec_from_file_location("build_training_report", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_training_report_uses_four_separate_timing_charts() -> None:
    template = REPORT_TEMPLATE.read_text(encoding="utf-8")

    for chart_id in (
        "step-timing-chart",
        "log-window-timing-chart",
        "checkpoint-duration-chart",
        "validation-duration-chart",
    ):
        assert f'id="{chart_id}"' in template
    assert 'id="event-duration-chart"' not in template


def test_parse_training_log_collects_training_domains_and_best_updates(tmp_path) -> None:
    log = tmp_path / "train.log"
    log.write_text("""2026-09-01 22:50:09,507 | INFO | trainer | epoch=1 step=10000 loss=2.125895 lr=2.7e-05 grad_norm=8.3 tokens=32907795 tokens_per_second=2642.9 progress=20.27% elapsed_seconds=12451 eta_seconds=48986 best_validation_loss=2.855694 peak_memory_mb=1650.0 gpu_memory_mb=1083.2/1856.0/3770.2 nonfinite_updates=0 log_interval_seconds=31.06 seconds_per_step=1.242 next_log_eta_seconds=31.1 next_checkpoint_eta_seconds=0.0 next_validation_eta_seconds=2484.0 (avg=2.325153)
2026-09-01 23:04:20,569 | INFO | trainer | validation_domain=english epoch=1 step=10000 loss=2.914236 cross_entropy=2.904576 perplexity=18.2575 tokens=1458682 batches=9129
2026-09-01 23:04:20,569 | INFO | trainer | validation epoch=1 step=10000 loss=2.811581 cross_entropy=2.802199 perplexity=16.4808 tokens=4930332 batches=22530
2026-09-01 23:04:20,569 | INFO | trainer | validation_timing step=10000 duration_seconds=851.06
2026-09-01 23:04:20,569 | INFO | trainer | new_best_validation step=10000 previous_loss=2.855694 loss=2.811581 metric=domains_v2
2026-09-01 23:04:21,100 | INFO | trainer | checkpoint kind=best step=10000 duration_seconds=0.53
""", encoding="utf-8")

    report = MODULE.parse_training_log(log)

    assert report["training"][0]["step"] == 10000
    assert report["training"][0]["progress"] == 20.27
    assert report["training"][0]["avg"] == 2.325153
    assert report["training"][0]["next_validation_eta_seconds"] == 2484.0
    assert report["training"][0]["session"] == 0
    assert report["validation"][0]["domains"]["english"]["loss"] == 2.914236
    assert report["best_updates"][0]["loss"] == 2.811581
    assert report["validation_timings"][0]["duration_seconds"] == 851.06
    assert report["checkpoint_timings"][0]["kind"] == "best"
    assert report["checkpoint_timings"][0]["duration_seconds"] == 0.53
    analysis = MODULE.analyze_progress(report)
    assert analysis["verdict"] == "waiting_for_validation"
    assert analysis["runtime"]["tokens_processed"] == 32907795
    assert analysis["run_summary"]["resume_count"] == 0


def test_atomic_json_output_is_valid(tmp_path) -> None:
    destination = tmp_path / "report.json"
    MODULE.write_atomic(destination, {"value": 1})
    assert json.loads(destination.read_text()) == {"value": 1}
    assert destination.read_text() == '{"value":1}\n'


def test_direct_script_does_not_shadow_stdlib_tokenize(tmp_path) -> None:
    import torch

    log = tmp_path / "train.log"
    checkpoint = tmp_path / "latest.pt"
    output = tmp_path / "report.json"
    log.write_text("2026 | INFO | trainer | epoch=1 step=1 loss=3.0\n")
    torch.save({"step": 1}, checkpoint)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--log", str(log),
            "--output", str(output),
            "--latest-checkpoint", str(checkpoint),
            "--best-checkpoint", str(tmp_path / "best.pt"),
            "--watch-seconds", "0",
        ],
        cwd=SCRIPT.parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["checkpoints"]["latest"]["step"] == 1


def test_incremental_reader_only_appends_new_complete_lines(tmp_path) -> None:
    log = tmp_path / "train.log"
    log.write_bytes(b"2026 | INFO | trainer | epoch=1 step=25 loss=3.0 tokens_per_second=10 progress=1%")
    reader = MODULE.IncrementalLogReader(log, raw_tail_lines=10)
    assert reader.refresh()["training"] == []
    with log.open("ab") as stream:
        stream.write(b"\n2026 | INFO | trainer | epoch=1 step=50 loss=2.8 tokens_per_second=11 progress=2%\n")

    parsed = reader.refresh()

    assert [item["step"] for item in parsed["training"]] == [25, 50]
    assert reader.refresh()["line_count"] == 2


def test_parser_marks_records_after_resume_as_a_new_session(tmp_path) -> None:
    log = tmp_path / "train.log"
    log.write_text(
        "2026 | INFO | trainer | epoch=1 step=100 loss=3 tokens_per_second=10 progress=2%\n"
        "2026 | INFO | app | Appending resumed training report data to train.log\n"
        "2026 | INFO | trainer | epoch=1 step=75 loss=2.9 tokens_per_second=10 progress=1.5%\n",
        encoding="utf-8",
    )

    parsed = MODULE.parse_training_log(log)

    assert [item["session"] for item in parsed["training"]] == [0, 1]
    assert parsed["session_count"] == 1


def test_parser_preserves_exact_run_configuration_snapshot(tmp_path) -> None:
    log = tmp_path / "train.log"
    snapshot = {
        "model_config": {"layers": 16},
        "training_config": {"validation_metric_name": "fixed-v5"},
    }
    log.write_text(
        "2026 | INFO | app | run_configuration="
        + json.dumps(snapshot, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    parsed = MODULE.parse_training_log(log)

    assert parsed["run_configurations"] == [{**snapshot, "session": 0}]


def test_normalize_history_sorts_and_replaces_restarted_steps() -> None:
    parsed = {
        "training": [
            {"epoch": 1, "step": 50, "loss": 3.0},
            {"epoch": 1, "step": 25, "loss": 3.2},
            {"epoch": 1, "step": 50, "loss": 2.8},
        ],
        "validation": [
            {"epoch": 1, "step": 100, "loss": 2.7},
            {"epoch": 1, "step": 100, "loss": 2.6},
        ],
        "best_updates": [
            {"step": 100, "loss": 2.7},
            {"step": 100, "loss": 2.6},
        ],
    }

    normalized = MODULE.normalize_history(parsed)

    assert [item["step"] for item in normalized["training"]] == [25, 50]
    assert normalized["training"][-1]["loss"] == 2.8
    assert normalized["validation"][0]["loss"] == 2.6
    assert normalized["best_updates"][0]["loss"] == 2.6


def test_normalize_history_removes_abandoned_future_after_resume() -> None:
    parsed = {
        "training": [
            {"epoch": 1, "step": 8000, "loss": 4.6},
            {"epoch": 1, "step": 8675, "loss": 4.45},
            {"epoch": 1, "step": 8025, "loss": 4.55},
            {"epoch": 1, "step": 8450, "loss": 4.52},
        ],
        "validation": [
            {"epoch": 1, "step": 8000, "loss": 4.53},
            {"epoch": 1, "step": 8500, "loss": 4.50},
        ],
        "best_updates": [
            {"step": 8000, "loss": 4.53},
            {"step": 8500, "loss": 4.50},
        ],
        "checkpoint_timings": [
            {"step": 8000, "duration_seconds": 4.0},
            {"step": 8500, "duration_seconds": 5.0},
        ],
        "validation_timings": [
            {"step": 8000, "duration_seconds": 10.0},
            {"step": 8500, "duration_seconds": 11.0},
        ],
    }

    normalized = MODULE.normalize_history(parsed)

    assert [item["step"] for item in normalized["training"]] == [8000, 8025, 8450]
    assert [item["step"] for item in normalized["validation"]] == [8000]
    assert [item["step"] for item in normalized["best_updates"]] == [8000]
    assert [item["step"] for item in normalized["checkpoint_timings"]] == [8000]
    assert [item["step"] for item in normalized["validation_timings"]] == [8000]
    assert normalized["resume_rollbacks"] == [{"epoch": 1, "step": 8025}]

    analysis = MODULE.analyze_progress(normalized)
    assert analysis["run_summary"]["resume_count"] == 0
    assert analysis["run_summary"]["rollback_count"] == 1


def test_progress_analysis_reports_overall_and_domain_improvement() -> None:
    parsed = {
        "training": [
            {"loss": 3.0, "avg": 3.1, "tokens_per_second": 10, "grad_norm": 2, "peak_memory_mb": 100},
            {"loss": 2.0, "avg": 2.5, "tokens_per_second": 12, "grad_norm": 3, "peak_memory_mb": 110, "tokens": 1000},
        ],
        "validation": [
            {"loss": 3.0, "perplexity": 20.0, "domains": {"chat": {"loss": 3.5, "perplexity": 25.0}}},
            {"loss": 2.7, "perplexity": 15.0, "domains": {"chat": {"loss": 3.0, "perplexity": 20.0}}},
        ],
    }
    analysis = MODULE.analyze_progress(parsed)
    assert analysis["verdict"] == "improving"
    assert analysis["overall_validation_loss"]["percent_improvement"] == pytest.approx(10.0)
    assert analysis["domains"]["chat"]["loss"]["absolute_improvement"] == 0.5
    assert analysis["overfitting"]["status"] == "no_current_signal"
    assert analysis["domain_ranking"][0]["name"] == "chat"
    assert analysis["checkpoint_comparison"]["best_step"] is None
    assert analysis["checkpoint_comparison"]["latest_minus_best"] is None
    assert analysis["checkpoint_comparison"]["status"] == "same_checkpoint"
    assert analysis["report_coverage"]["generation_quality"].startswith("pending")


def test_progress_analysis_detects_overfitting_signal() -> None:
    parsed = {
        "training": [{"loss": 3.0}, {"loss": 2.0}],
        "validation": [{"step": 10, "loss": 2.5}, {"step": 20, "loss": 2.7}],
        "best_updates": [],
    }

    analysis = MODULE.analyze_progress(parsed)

    assert analysis["overfitting"]["status"] == "risk_detected"
    assert analysis["checkpoint_comparison"]["best_step"] == 10
    assert analysis["checkpoint_comparison"]["latest_minus_best"] == pytest.approx(0.2)
    assert analysis["checkpoint_comparison"]["status"] == "different_checkpoints"


def test_progress_analysis_excludes_incompatible_validation_protocols() -> None:
    parsed = {
        "training": [],
        "validation": [
            {"step": 10, "loss": 1.0, "metric": "old"},
            {"step": 20, "loss": 3.0, "metric": "new"},
            {"step": 30, "loss": 2.5, "metric": "new"},
        ],
        "best_updates": [],
    }

    analysis = MODULE.analyze_progress(parsed)

    assert analysis["overall_validation_loss"]["first"] == 3.0
    assert analysis["overall_validation_loss"]["latest"] == 2.5
    assert analysis["run_summary"]["active_validation_metric"] == "new"
    assert analysis["run_summary"]["excluded_incompatible_validations"] == 1


def test_checkpoint_details_only_attaches_matching_validation(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"placeholder")
    monkeypatch.setattr(MODULE, "_checkpoint_step", lambda path: 20)

    details = MODULE.checkpoint_details(
        checkpoint,
        [{"step": 10, "loss": 1.0}, {"step": 20, "loss": 2.0}],
        best=True,
    )

    assert details["step"] == 20
    assert details["validation"] == {"step": 20, "loss": 2.0}


def test_evaluation_artifacts_update_report_coverage(tmp_path) -> None:
    audit = tmp_path / "audit.json"
    benchmark = tmp_path / "benchmark.json"
    audit.write_text('{"status":"passed","findings":[]}')
    benchmark.write_text('{"summary":{"accuracy":0.75},"results":[]}')

    coverage = MODULE.evaluation_coverage(
        MODULE.load_evaluation_artifact(audit),
        MODULE.load_evaluation_artifact(benchmark),
    )

    assert coverage["data_quality"] == "available (passed)"
    assert coverage["generation_quality"] == "available (accuracy: 75.0%)"


def test_invalid_evaluation_artifact_is_treated_as_missing(tmp_path) -> None:
    artifact = tmp_path / "invalid.json"
    artifact.write_text("not json")

    assert MODULE.load_evaluation_artifact(artifact) is None


def test_report_output_cannot_be_loaded_as_its_own_evaluation(tmp_path) -> None:
    report = tmp_path / "report.json"
    report.write_text('{"evaluations":{"generation_quality":{"summary":{"accuracy":1}}}}')

    assert MODULE.load_evaluation_artifact(report, forbidden_path=report) is None


def test_pid_check_accepts_a_live_process_and_rejects_missing_process() -> None:
    assert MODULE._pid_is_running(os.getpid()) is True
    assert MODULE._pid_is_running(2**31 - 1) is False


def test_system_monitor_collects_cpu_ram_and_process_memory() -> None:
    monitor = MODULE.SystemMonitor(os.getpid(), max_points=2)
    monitor.sample()
    sample = monitor.sample()

    assert sample["cpu_percent"] is not None
    assert 0 <= sample["cpu_percent"] <= 100
    assert sample["ram_total_mb"] > 0
    assert sample["ram_used_mb"] >= 0
    assert sample["process_rss_mb"] > 0
    assert len(monitor.history) == 2


def test_cpu_temperature_prefers_package_sensor(tmp_path) -> None:
    hwmon = tmp_path / "hwmon"
    device = hwmon / "hwmon0"
    device.mkdir(parents=True)
    (device / "name").write_text("coretemp\n")
    (device / "temp1_input").write_text("52000\n")
    (device / "temp1_label").write_text("Core 0\n")
    (device / "temp2_input").write_text("61000\n")
    (device / "temp2_label").write_text("Package id 0\n")

    temperature = MODULE.SystemMonitor._cpu_temperature_c(hwmon, tmp_path / "thermal")

    assert temperature == 61


def test_gpu_monitor_parses_nvidia_smi_and_handles_na(monkeypatch) -> None:
    output = "0, NVIDIA RTX, 72, 1906, 4096, 63, 22.5, [N/A], 60, 75, [N/A]\n"
    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=output),
    )

    gpu = MODULE.SystemMonitor._gpus()[0]

    assert gpu["name"] == "NVIDIA RTX"
    assert gpu["utilization_percent"] == 72
    assert gpu["memory_used_mb"] == 1906
    assert gpu["temperature_c"] == 63
    assert gpu["power_draw_w"] == 22.5
    assert gpu["power_limit_w"] == 60
    assert gpu["power_enforced_limit_w"] is None
    assert gpu["power_default_limit_w"] == 60
    assert gpu["power_max_limit_w"] == 75
    assert gpu["fan_percent"] is None


@pytest.mark.parametrize('accuracy', [True, False, '0.75', -0.1, 1.1, float('nan'), float('inf'), None])
def test_generation_accuracy_rejects_invalid_scores(accuracy) -> None:
    artifact = {'summary': {'accuracy': accuracy}}
    assert MODULE.generation_accuracy(artifact) is None
    assert MODULE.evaluation_coverage(None, artifact)['generation_quality'].startswith('pending')


@pytest.mark.parametrize('accuracy', [0.0, 0.75, 1.0])
def test_generation_accuracy_accepts_proportions(accuracy) -> None:
    assert MODULE.generation_accuracy({'summary': {'accuracy': accuracy}}) == accuracy


def test_generation_quality_distinguishes_probes_and_empty_benchmarks() -> None:
    assert MODULE.generation_accuracy({'summary': {'cases': 0, 'accuracy': 0}}) is None
    coverage = MODULE.evaluation_coverage(None, {'responses': [{'prompt': 'Hello', 'response': 'Hi'}]})
    assert coverage['generation_quality'] == 'available (qualitative probes; accuracy not measured)'


def test_model_deployment_analysis_reports_hybrid_memory_alternatives():
    from utils.config import load_yaml
    analysis = MODULE.model_deployment_analysis(
        load_yaml("configs/model.hybrid.gpu.yaml"),
        {"max_sequence_length": 4096},
        [{"gpus": [{"memory_total_mb": 4096}]}],
    )
    assert analysis["available"] is True
    assert analysis["attention_pattern"] == "hybrid"
    assert analysis["linear_attention_layers"] == 12
    assert analysis["full_attention_layers"] == 4
    assert analysis["memory_budget_gib"] == pytest.approx(4.0)
    assert len(analysis["deployment_matrix"]) == 12
    assert analysis["deployment_matrix"][0]["estimated_total_bytes"] <= analysis["deployment_matrix"][-1]["estimated_total_bytes"]


def test_parser_captures_mtp_and_moe_auxiliary_losses(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(
        "2026 | INFO | trainer | epoch=1 step=1 loss=2.0 lr=1e-4 grad_norm=1 "
        "tokens=10 tokens_per_second=5 progress=1% mtp_loss=0.25 moe_aux_loss=0.01 (avg=2.0)\n",
        encoding="utf-8",
    )
    parsed = MODULE.parse_training_log(log)
    assert parsed["training"][0]["mtp_loss"] == pytest.approx(0.25)
    assert parsed["training"][0]["moe_aux_loss"] == pytest.approx(0.01)


def test_progress_analysis_includes_training_and_validation_timestamps() -> None:
    parsed = {
        "training": [
            {"timestamp": "2026-09-01 10:00:00,000", "loss": 3.0, "step": 10},
            {"timestamp": "2026-09-01 10:30:00,000", "loss": 2.0, "step": 20},
        ],
        "validation": [
            {"timestamp": "2026-09-01 10:15:00,000", "loss": 2.8, "step": 10},
            {"timestamp": "2026-09-01 10:35:00,000", "loss": 2.5, "step": 20},
        ],
        "best_updates": [],
    }
    analysis = MODULE.analyze_progress(parsed)
    assert analysis["run_summary"]["training_start_time"] == "2026-09-01 10:00:00,000"
    assert analysis["run_summary"]["training_end_time"] == "2026-09-01 10:30:00,000"
    assert analysis["run_summary"]["validation_start_time"] == "2026-09-01 10:15:00,000"
    assert analysis["run_summary"]["validation_end_time"] == "2026-09-01 10:35:00,000"
    assert analysis["runtime"]["training_start_time"] == "2026-09-01 10:00:00,000"
    assert analysis["runtime"]["training_end_time"] == "2026-09-01 10:30:00,000"
    assert analysis["runtime"]["validation_start_time"] == "2026-09-01 10:15:00,000"
    assert analysis["runtime"]["validation_end_time"] == "2026-09-01 10:35:00,000"


def test_parser_captures_generation_timings(tmp_path) -> None:
    log = tmp_path / "train.log"
    log.write_text(
        "2026-09-01 10:00:00,000 | INFO | trainer | generation_evaluation epoch=1 step=50 accuracy=0.8500 cases=20 duration_seconds=12.34\n",
        encoding="utf-8",
    )
    parsed = MODULE.parse_training_log(log)
    assert len(parsed["generation_timings"]) == 1
    assert parsed["generation_timings"][0]["step"] == 50
    assert parsed["generation_timings"][0]["accuracy"] == 0.85
    assert parsed["generation_timings"][0]["duration_seconds"] == 12.34
    analysis = MODULE.analyze_progress(parsed)
    assert analysis["runtime"]["latest_generation_duration_seconds"] == 12.34


def test_parser_captures_validation_events_and_active_state(tmp_path) -> None:
    log = tmp_path / "train.log"
    log.write_text(
        "2026-09-01 10:00:00,000 | INFO | trainer | epoch=1 step=100 loss=2.0 tokens_per_second=1000 progress=50%\n"
        "2026-09-01 10:00:05,000 | INFO | trainer | validation_started step=100\n"
        "2026-09-01 10:00:10,000 | INFO | evaluator | validation_progress name=general batches=20/50 elapsed_seconds=5.0 eta_seconds=7.5\n",
        encoding="utf-8",
    )
    parsed = MODULE.parse_training_log(log)
    assert len(parsed["validation_events"]) == 2
    assert parsed["validation_events"][0]["event"] == "started"
    assert parsed["validation_events"][1]["event"] == "progress"
    assert parsed["validation_events"][1]["batches"] == "20/50"
    assert parsed["validation_events"][1]["elapsed_seconds"] == 5.0
    assert parsed["validation_events"][1]["eta_seconds"] == 7.5

    analysis = MODULE.analyze_progress(parsed)
    active = analysis["runtime"]["active_validation"]
    assert active["running"] is True
    assert active["step"] == 100
    assert active["name"] == "general"
    assert active["batches"] == "20/50"
    assert active["elapsed_seconds"] == 5.0
    assert active["eta_seconds"] == 7.5

    # Now append completion
    with log.open("a", encoding="utf-8") as f:
        f.write("2026-09-01 10:00:18,000 | INFO | trainer | validation epoch=1 step=100 loss=1.9 cross_entropy=1.85 perplexity=6.36 tokens=50000 batches=50\n")
        f.write("2026-09-01 10:00:18,000 | INFO | trainer | validation_timing step=100 duration_seconds=13.0\n")

    parsed_done = MODULE.parse_training_log(log)
    analysis_done = MODULE.analyze_progress(parsed_done)
    assert analysis_done["runtime"]["active_validation"]["running"] is False
    assert analysis_done["runtime"]["latest_validation_duration_seconds"] == 13.0
    assert analysis_done["runtime"]["average_validation_duration_seconds"] == 13.0


