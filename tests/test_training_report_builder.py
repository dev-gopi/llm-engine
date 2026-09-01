import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_training_report.py"
SPEC = importlib.util.spec_from_file_location("build_training_report", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_parse_training_log_collects_training_domains_and_best_updates(tmp_path) -> None:
    log = tmp_path / "train.log"
    log.write_text("""2026-09-01 22:50:09,507 | INFO | trainer | epoch=1 step=10000 loss=2.125895 lr=2.7e-05 grad_norm=8.3 tokens=32907795 tokens_per_second=2642.9 progress=20.27% elapsed_seconds=12451 eta_seconds=48986 best_validation_loss=2.855694 peak_memory_mb=1650.0 gpu_memory_mb=1083.2/1856.0/3770.2 nonfinite_updates=0 (avg=2.325153)
2026-09-01 23:04:20,569 | INFO | trainer | validation_domain=english epoch=1 step=10000 loss=2.914236 cross_entropy=2.904576 perplexity=18.2575 tokens=1458682 batches=9129
2026-09-01 23:04:20,569 | INFO | trainer | validation epoch=1 step=10000 loss=2.811581 cross_entropy=2.802199 perplexity=16.4808 tokens=4930332 batches=22530
2026-09-01 23:04:20,569 | INFO | trainer | new_best_validation step=10000 previous_loss=2.855694 loss=2.811581 metric=domains_v2
""", encoding="utf-8")

    report = MODULE.parse_training_log(log)

    assert report["training"][0]["step"] == 10000
    assert report["training"][0]["progress"] == 20.27
    assert report["training"][0]["avg"] == 2.325153
    assert report["validation"][0]["domains"]["english"]["loss"] == 2.914236
    assert report["best_updates"][0]["loss"] == 2.811581
    analysis = MODULE.analyze_progress(report)
    assert analysis["verdict"] == "waiting_for_validation"
    assert analysis["runtime"]["tokens_processed"] == 32907795


def test_atomic_json_output_is_valid(tmp_path) -> None:
    destination = tmp_path / "report.json"
    MODULE.write_atomic(destination, {"value": 1})
    assert json.loads(destination.read_text()) == {"value": 1}
    assert destination.read_text() == '{"value":1}\n'


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
    assert analysis["checkpoint_comparison"]["latest_minus_best"] == 0
    assert analysis["report_coverage"]["generation_quality"].startswith("not_collected")


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


def test_gpu_monitor_parses_nvidia_smi_and_handles_na(monkeypatch) -> None:
    output = "0, NVIDIA RTX, 72, 1906, 4096, 63, 22.5, 60, [N/A]\n"
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
    assert gpu["fan_percent"] is None
