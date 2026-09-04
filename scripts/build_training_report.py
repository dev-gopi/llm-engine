"""Build or continuously refresh standalone JSON data for the training report viewer."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any

script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path[:] = [entry for entry in sys.path if os.path.realpath(entry or ".") != script_directory]

from utils.config import load_yaml


KEY_VALUE = re.compile(r"([a-zA-Z_]+)=([^\s]+)")


def _number(value: str) -> float | None:
    try:
        parsed = float(value.strip())
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def load_evaluation_artifact(path: Path | None) -> dict[str, Any] | None:
    """Load an optional evaluation result without breaking the live reporter."""
    if path is None or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def evaluation_coverage(
    data_audit: dict[str, Any] | None,
    generation_evaluation: dict[str, Any] | None,
) -> dict[str, str]:
    data_status = "not_collected; run a dataset audit for duplicates, truncation, language and token lengths"
    if data_audit:
        status = str(data_audit.get("status", "available"))
        data_status = f"available ({status})"
    generation_status = "pending; no fixed-prompt or benchmark result is available"
    if generation_evaluation:
        summary = generation_evaluation.get("summary")
        accuracy = summary.get("accuracy") if isinstance(summary, dict) else None
        generation_status = "available" + (
            f" (accuracy: {float(accuracy):.1%})" if isinstance(accuracy, (int, float)) else ""
        )
    return {"data_quality": data_status, "generation_quality": generation_status}


class SystemMonitor:
    """Collect lightweight host and NVIDIA telemetry outside the trainer."""

    def __init__(self, process_pid: int | None = None, *, max_points: int = 3600) -> None:
        self.process_pid = process_pid
        self.max_points = max_points
        self.previous_cpu: tuple[int, int] | None = None
        self.history: list[dict[str, Any]] = []

    def sample(self) -> dict[str, Any]:
        sample: dict[str, Any] = {"timestamp": datetime.now(timezone.utc).isoformat()}
        sample.update(self._cpu_and_memory())
        sample["cpu_temperature_c"] = self._cpu_temperature_c()
        sample["process_rss_mb"] = self._process_rss_mb()
        sample["gpus"] = self._gpus()
        self.history.append(sample)
        del self.history[:-self.max_points]
        return sample

    def _cpu_and_memory(self) -> dict[str, Any]:
        result = {"cpu_percent": None, "ram_used_mb": None, "ram_total_mb": None, "ram_percent": None}
        try:
            cpu_values = [int(value) for value in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
            total, idle = sum(cpu_values), cpu_values[3] + (cpu_values[4] if len(cpu_values) > 4 else 0)
            if self.previous_cpu:
                total_delta, idle_delta = total - self.previous_cpu[0], idle - self.previous_cpu[1]
                if total_delta > 0:
                    result["cpu_percent"] = (1.0 - idle_delta / total_delta) * 100.0
            self.previous_cpu = (total, idle)
            memory = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, value = line.split(":", 1)
                memory[key] = int(value.strip().split()[0])
            total_kb, available_kb = memory["MemTotal"], memory["MemAvailable"]
            used_kb = total_kb - available_kb
            result.update({
                "ram_used_mb": used_kb / 1024,
                "ram_total_mb": total_kb / 1024,
                "ram_percent": used_kb / total_kb * 100 if total_kb else None,
            })
        except (OSError, ValueError, KeyError, IndexError):
            pass
        return result

    @staticmethod
    def _cpu_temperature_c(
        hwmon_root: Path = Path("/sys/class/hwmon"),
        thermal_root: Path = Path("/sys/class/thermal"),
    ) -> float | None:
        """Return the best available CPU package temperature on Linux."""
        candidates: list[tuple[int, float]] = []
        try:
            for device in hwmon_root.glob("hwmon*"):
                try:
                    name = (device / "name").read_text().strip().lower()
                except OSError:
                    continue
                if not any(token in name for token in ("coretemp", "k10temp", "zenpower", "cpu")):
                    continue
                for sensor in device.glob("temp*_input"):
                    try:
                        value = float(sensor.read_text().strip()) / 1000.0
                        label_path = sensor.with_name(sensor.name.replace("_input", "_label"))
                        label = label_path.read_text().strip().lower() if label_path.exists() else ""
                    except (OSError, ValueError):
                        continue
                    if not -20 <= value <= 150:
                        continue
                    priority = 0 if any(token in label for token in ("package", "tctl", "tdie")) else 1
                    candidates.append((priority, value))
        except OSError:
            pass
        if candidates:
            best_priority = min(priority for priority, _ in candidates)
            return max(value for priority, value in candidates if priority == best_priority)
        try:
            for zone in thermal_root.glob("thermal_zone*"):
                try:
                    kind = (zone / "type").read_text().strip().lower()
                    value = float((zone / "temp").read_text().strip()) / 1000.0
                except (OSError, ValueError):
                    continue
                if any(token in kind for token in ("cpu", "package", "x86_pkg", "soc")) and -20 <= value <= 150:
                    return value
        except OSError:
            pass
        return None

    def _process_rss_mb(self) -> float | None:
        if not self.process_pid:
            return None
        try:
            for line in Path(f"/proc/{self.process_pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
        except (OSError, ValueError, IndexError):
            pass
        return None

    @staticmethod
    def _gpus() -> list[dict[str, Any]]:
        fields = [
            "index", "name", "utilization.gpu", "memory.used", "memory.total",
            "temperature.gpu", "power.draw", "power.limit", "power.default_limit",
            "power.max_limit", "fan.speed",
        ]
        try:
            completed = subprocess.run(
                ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return []
        if completed.returncode != 0:
            return []
        gpus = []
        for line in completed.stdout.splitlines():
            values = [value.strip() for value in line.split(",")]
            if len(values) != len(fields):
                continue
            enforced_limit = _number(values[7])
            default_limit = _number(values[8])
            maximum_limit = _number(values[9])
            gpus.append({
                "index": int(values[0]), "name": values[1],
                "utilization_percent": _number(values[2]),
                "memory_used_mb": _number(values[3]), "memory_total_mb": _number(values[4]),
                "temperature_c": _number(values[5]), "power_draw_w": _number(values[6]),
                "power_limit_w": enforced_limit or default_limit or maximum_limit,
                "power_enforced_limit_w": enforced_limit,
                "power_default_limit_w": default_limit,
                "power_max_limit_w": maximum_limit,
                "fan_percent": _number(values[10]),
            })
        return gpus


def _value(text: str) -> Any:
    text = text.strip("(),%")
    if text in {"inf", "+inf", "-inf", "nan"}:
        return None
    try:
        return float(text) if any(char in text for char in ".eE") else int(text)
    except ValueError:
        return text


def _fields(message: str) -> dict[str, Any]:
    return {key: _value(value) for key, value in KEY_VALUE.findall(message)}


def _empty_parsed() -> dict[str, Any]:
    return {
        "training": [], "validation": [], "best_updates": [], "warnings": [],
        "raw_log_tail": [], "line_count": 0, "session_count": 0,
    }


def _append_lines(
    parsed: dict[str, Any],
    lines: list[str],
    pending_domains: dict[tuple[int, int], dict[str, dict[str, Any]]],
    *,
    raw_tail_lines: int,
) -> None:
    for line in lines:
        parsed["line_count"] += 1
        message = line.rsplit(" | ", 1)[-1]
        timestamp = line.split(" | ", 1)[0] if " | " in line else None
        if "Appending resumed training report data" in message:
            parsed["session_count"] += 1
        elif "validation_domain=" in message:
            values = _fields(message)
            key = (int(values.get("epoch", 0)), int(values.get("step", 0)))
            domain = str(values.pop("validation_domain"))
            values["timestamp"] = timestamp
            values["session"] = parsed["session_count"]
            pending_domains.setdefault(key, {})[domain] = values
        elif message.startswith("validation epoch="):
            values = _fields(message)
            key = (int(values.get("epoch", 0)), int(values.get("step", 0)))
            values["timestamp"] = timestamp
            values["session"] = parsed["session_count"]
            values["domains"] = pending_domains.pop(key, {})
            parsed["validation"].append(values)
        elif "new_best_validation" in message:
            values = _fields(message)
            values["timestamp"] = timestamp
            values["session"] = parsed["session_count"]
            parsed["best_updates"].append(values)
        elif "epoch=" in message and "tokens_per_second=" in message and "loss=" in message:
            values = _fields(message)
            values["timestamp"] = timestamp
            values["session"] = parsed["session_count"]
            parsed["training"].append(values)
        elif "WARNING" in line or "ERROR" in line:
            parsed["warnings"].append(line)
    if raw_tail_lines > 0:
        parsed["raw_log_tail"].extend(lines)
        del parsed["raw_log_tail"][:-raw_tail_lines]


def parse_training_log(path: str | Path, *, raw_tail_lines: int = 1000) -> dict[str, Any]:
    """Parse trainer INFO lines without importing or interacting with training code."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    parsed = _empty_parsed()
    _append_lines(parsed, lines, {}, raw_tail_lines=raw_tail_lines)
    return parsed


def normalize_history(parsed: dict[str, Any]) -> dict[str, Any]:
    """Keep the active timeline when a resumed run rolls back to a checkpoint.

    A resume can append to a log whose previous process had advanced beyond the
    last durable checkpoint.  Merely sorting/deduplicating those records leaves
    the abandoned future at the end of the report, making every headline value
    stale until the resumed process catches up.
    """
    normalized = dict(parsed)

    training_records: dict[tuple[int, int], dict[str, Any]] = {}
    rollback_ranges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    latest_position: tuple[int, int] | None = None
    for item in parsed.get("training", []):
        position = (int(item.get("epoch", 0)), int(item.get("step", 0)))
        if latest_position is not None and position < latest_position:
            rollback_ranges.append((position, latest_position))
            training_records = {
                key: value for key, value in training_records.items() if key < position
            }
        training_records[position] = item
        latest_position = position
    normalized["training"] = [training_records[key] for key in sorted(training_records)]

    validation_records: dict[tuple[int, int], dict[str, Any]] = {}
    for item in parsed.get("validation", []):
        position = (int(item.get("epoch", 0)), int(item.get("step", 0)))
        if any(start < position <= abandoned_end for start, abandoned_end in rollback_ranges):
            continue
        validation_records[position] = item
    normalized["validation"] = [validation_records[key] for key in sorted(validation_records)]

    best_updates: dict[int, dict[str, Any]] = {}
    for item in parsed.get("best_updates", []):
        step = int(item.get("step", 0))
        if any(start[1] < step <= abandoned_end[1] for start, abandoned_end in rollback_ranges):
            continue
        best_updates[step] = item
    normalized["best_updates"] = [best_updates[key] for key in sorted(best_updates)]
    normalized["resume_rollbacks"] = [
        {"epoch": start[0], "step": start[1]} for start, _ in rollback_ranges
    ]
    return normalized


class IncrementalLogReader:
    """Tail only newly appended bytes and retain parsed history in this process."""

    def __init__(self, path: Path, *, raw_tail_lines: int) -> None:
        self.path = path
        self.raw_tail_lines = raw_tail_lines
        self.offset = 0
        self.partial = b""
        self.parsed = _empty_parsed()
        self.pending_domains: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}

    def refresh(self) -> dict[str, Any]:
        size = self.path.stat().st_size
        if size < self.offset:
            # A restarted run truncated or replaced the log.
            self.offset = 0
            self.partial = b""
            self.parsed = _empty_parsed()
            self.pending_domains = {}
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            chunk = stream.read()
        self.offset += len(chunk)
        if not chunk:
            return self.parsed
        pieces = (self.partial + chunk).split(b"\n")
        self.partial = pieces.pop()
        lines = [piece.rstrip(b"\r").decode("utf-8", errors="replace") for piece in pieces]
        _append_lines(
            self.parsed, lines, self.pending_domains,
            raw_tail_lines=self.raw_tail_lines,
        )
        return self.parsed


def checkpoint_details(path: str | Path, validation: list[dict[str, Any]], *, best: bool) -> dict[str, Any]:
    checkpoint = Path(path)
    details: dict[str, Any] = {"path": str(checkpoint), "exists": checkpoint.is_file()}
    if checkpoint.is_file():
        stat = checkpoint.stat()
        details.update({
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / 1024**2, 2),
            "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        })
    if validation:
        selected = min(validation, key=lambda item: float(item["loss"])) if best else validation[-1]
        details["validation"] = selected
    return details


def _change(first: float | int | None, latest: float | int | None) -> dict[str, Any]:
    if first is None or latest is None:
        return {"first": first, "latest": latest, "absolute_improvement": None, "percent_improvement": None}
    improvement = float(first) - float(latest)
    percent = improvement / abs(float(first)) * 100.0 if float(first) else None
    return {
        "first": first,
        "latest": latest,
        "absolute_improvement": improvement,
        "percent_improvement": percent,
    }


def analyze_progress(parsed: dict[str, Any]) -> dict[str, Any]:
    """Calculate lower-is-better improvements and runtime health summaries."""
    training, validation = parsed["training"], parsed["validation"]
    overall = _change(
        validation[0].get("loss") if validation else None,
        validation[-1].get("loss") if validation else None,
    )
    recent = _change(
        validation[-2].get("loss") if len(validation) > 1 else None,
        validation[-1].get("loss") if validation else None,
    )
    perplexity = _change(
        validation[0].get("perplexity") if validation else None,
        validation[-1].get("perplexity") if validation else None,
    )
    domains: dict[str, Any] = {}
    domain_names = sorted({name for item in validation for name in item.get("domains", {})})
    for name in domain_names:
        records = [item["domains"][name] for item in validation if name in item.get("domains", {})]
        domains[name] = {
            "loss": _change(records[0].get("loss"), records[-1].get("loss")),
            "perplexity": _change(records[0].get("perplexity"), records[-1].get("perplexity")),
            "observations": len(records),
        }
    improvement = recent.get("absolute_improvement")
    if improvement is None:
        verdict = "waiting_for_validation"
    elif improvement >= 0.001:
        verdict = "improving"
    elif improvement > -0.001:
        verdict = "plateau"
    else:
        verdict = "worsening"
    throughput = [float(item["tokens_per_second"]) for item in training if item.get("tokens_per_second") is not None]
    gradients = [float(item["grad_norm"]) for item in training if item.get("grad_norm") is not None]
    memory = [float(item["peak_memory_mb"]) for item in training if item.get("peak_memory_mb") is not None]
    train_metric = "avg" if training and training[0].get("avg") is not None else "loss"
    training_loss = _change(
        training[0].get(train_metric) if training else None,
        training[-1].get(train_metric) if training else None,
    )
    domain_ranking = sorted(
        (
            {"name": name, "latest_loss": values["loss"]["latest"],
             "percent_improvement": values["loss"]["percent_improvement"]}
            for name, values in domains.items()
            if values["loss"]["latest"] is not None
        ),
        key=lambda item: item["latest_loss"],
    )
    train_improvement = training_loss.get("absolute_improvement")
    recent_validation_improvement = recent.get("absolute_improvement")
    if len(validation) < 2:
        overfitting = {"status": "insufficient_validation", "reason": "At least two validation runs are required."}
    elif train_improvement is not None and train_improvement > 0 and recent_validation_improvement is not None and recent_validation_improvement < -0.001:
        overfitting = {"status": "risk_detected", "reason": "Training loss improved while recent validation loss worsened."}
    elif recent_validation_improvement is not None and recent_validation_improvement >= -0.001:
        overfitting = {"status": "no_current_signal", "reason": "Recent validation loss is stable or improving."}
    else:
        overfitting = {"status": "inconclusive", "reason": "The available loss trends are inconclusive."}
    best_validation = min(validation, key=lambda item: float(item["loss"])) if validation else None
    latest_validation = validation[-1] if validation else None
    same_checkpoint = bool(
        latest_validation
        and best_validation
        and latest_validation.get("step") == best_validation.get("step")
    )
    return {
        "verdict": verdict,
        "overall_validation_loss": overall,
        "recent_validation_loss": recent,
        "overall_perplexity": perplexity,
        "training_loss": training_loss,
        "domains": domains,
        "domain_ranking": domain_ranking,
        "overfitting": overfitting,
        "checkpoint_comparison": {
            "best_step": best_validation.get("step") if best_validation else None,
            "best_validation_loss": best_validation.get("loss") if best_validation else None,
            "latest_step": latest_validation.get("step") if latest_validation else None,
            "latest_validation_loss": latest_validation.get("loss") if latest_validation else None,
            "latest_minus_best": (
                float(latest_validation["loss"]) - float(best_validation["loss"])
                if latest_validation and best_validation and not same_checkpoint else None
            ),
            "status": "same_checkpoint" if same_checkpoint else "different_checkpoints",
            "generation_accuracy": None,
            "note": "Loss comparison only; response quality requires a separate fixed-prompt or benchmark evaluation.",
        },
        "run_summary": {
            "current_epoch": training[-1].get("epoch") if training else None,
            "current_step": training[-1].get("step") if training else None,
            "progress_percent": training[-1].get("progress") if training else None,
            "resume_count": int(parsed.get("session_count", 0)),
            "rollback_count": len(parsed.get("resume_rollbacks", [])),
            "best_validation_step": best_validation.get("step") if best_validation else None,
            "best_validation_loss": best_validation.get("loss") if best_validation else None,
            "best_checkpoint_updates": len(parsed.get("best_updates", [])),
        },
        "report_coverage": {
            "loss_and_runtime": "available",
            "checkpoint_files": "available",
            "data_quality": "not_collected; run a dataset audit for duplicates, truncation, language and token lengths",
            "generation_quality": "pending; no fixed-prompt or benchmark result is available",
            "gpu_telemetry": "partial; memory and throughput are available, utilization, temperature and power are not logged",
        },
        "runtime": {
            "training_points": len(training),
            "validation_points": len(validation),
            "average_tokens_per_second": sum(throughput) / len(throughput) if throughput else None,
            "minimum_tokens_per_second": min(throughput) if throughput else None,
            "maximum_tokens_per_second": max(throughput) if throughput else None,
            "average_gradient_norm": sum(gradients) / len(gradients) if gradients else None,
            "maximum_gradient_norm": max(gradients) if gradients else None,
            "peak_memory_mb": max(memory) if memory else None,
            "nonfinite_updates": training[-1].get("nonfinite_updates") if training else None,
            "current_learning_rate": training[-1].get("lr") if training else None,
            "tokens_processed": training[-1].get("tokens") if training else None,
            "elapsed_seconds": training[-1].get("elapsed_seconds") if training else None,
            "eta_seconds": training[-1].get("eta_seconds") if training else None,
        },
    }


def build_report(
    args: argparse.Namespace,
    parsed: dict[str, Any] | None = None,
    telemetry: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    parsed = normalize_history(
        parsed or parse_training_log(args.log, raw_tail_lines=args.raw_tail_lines)
    )
    data_audit = load_evaluation_artifact(getattr(args, "data_audit", None))
    generation_evaluation = load_evaluation_artifact(
        getattr(args, "generation_evaluation", None)
    )
    coverage = evaluation_coverage(data_audit, generation_evaluation)
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_log": str(args.log),
        "model_config_path": str(args.model_config),
        "training_config_path": str(args.training_config),
        "monitoring": {
            "telemetry_seconds": args.telemetry_seconds,
            "refresh_seconds": args.watch_seconds,
        },
        "model_config": load_yaml(args.model_config),
        "training_config": load_yaml(args.training_config),
        "checkpoints": {
            "latest": checkpoint_details(args.latest_checkpoint, parsed["validation"], best=False),
            "best": checkpoint_details(args.best_checkpoint, parsed["validation"], best=True),
        },
        "evaluations": {
            "data_quality": data_audit,
            "generation_quality": generation_evaluation,
        },
        "analysis": analyze_progress(parsed),
        "telemetry": telemetry or [],
        **parsed,
    }
    report["analysis"]["report_coverage"].update(coverage)
    generation_summary = (generation_evaluation or {}).get("summary")
    if isinstance(generation_summary, dict):
        accuracy = generation_summary.get("accuracy")
        comparison = report["analysis"]["checkpoint_comparison"]
        comparison["generation_accuracy"] = accuracy
        comparison["note"] = "Fixed-prompt benchmark results are included. Compare multiple checkpoint artifacts before deployment."
    latest_telemetry = report["telemetry"][-1] if report["telemetry"] else {}
    if latest_telemetry.get("gpus"):
        report["analysis"]["report_coverage"]["gpu_telemetry"] = "available"
    elif report["telemetry"]:
        report["analysis"]["report_coverage"]["gpu_telemetry"] = "CPU and RAM available; NVIDIA telemetry unavailable"
    return _finite(report)


async def watch_report(args: argparse.Namespace) -> None:
    """Refresh JSON asynchronously while file reads run off the event loop."""
    reader = IncrementalLogReader(args.log, raw_tail_lines=args.raw_tail_lines)
    monitor = SystemMonitor(args.parent_pid, max_points=args.telemetry_points)
    interval = max(args.watch_seconds, 0.25)
    last_line_count = -1
    last_telemetry_at = 0.0
    while True:
        if args.parent_pid and not _pid_is_running(args.parent_pid):
            print("parent training process exited; report watcher stopping", flush=True)
            return
        parsed = await asyncio.to_thread(reader.refresh)
        now = time.monotonic()
        telemetry_due = not monitor.history or now - last_telemetry_at >= args.telemetry_seconds
        log_changed = parsed["line_count"] != last_line_count
        if telemetry_due:
            await asyncio.to_thread(monitor.sample)
            last_telemetry_at = now
        if not log_changed and not telemetry_due:
            await asyncio.sleep(interval)
            continue
        report = await asyncio.to_thread(build_report, args, parsed, monitor.history)
        await asyncio.to_thread(write_atomic, args.output, report)
        last_line_count = parsed["line_count"]
        print(
            f"updated {args.output}: {len(report['training'])} training points, "
            f"{len(report['validation'])} validations",
            flush=True,
        )
        await asyncio.sleep(interval)


def _pid_is_running(pid: int) -> bool:
    """Return whether a process exists, including when the watcher was restarted."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    return value


def write_atomic(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            # Compact output substantially reduces disk writes during long live runs.
            # The dashboard parses JSON directly, so whitespace provides no value here.
            json.dump(
                report, stream, ensure_ascii=False, allow_nan=False,
                separators=(",", ":"),
            )
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log", type=Path, default=Path("logs/training.log"),
        help="training console log to parse (matches train.py default)",
    )
    parser.add_argument("--output", type=Path, default=Path("reports/training_report.json"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.gpu.yaml"))
    parser.add_argument("--training-config", type=Path, default=Path("configs/finetuning.gpu.yaml"))
    parser.add_argument("--latest-checkpoint", type=Path, default=Path("checkpoints/finetuning/latest.pt"))
    parser.add_argument("--best-checkpoint", type=Path, default=Path("checkpoints/finetuning/best.pt"))
    parser.add_argument(
        "--data-audit", type=Path, default=Path("reports/data_quality.json"),
        help="optional JSON dataset-quality audit included in report coverage",
    )
    parser.add_argument(
        "--generation-evaluation", type=Path, default=Path("reports/generation_quality.json"),
        help="optional JSON fixed-prompt/benchmark result included in report coverage",
    )
    parser.add_argument("--raw-tail-lines", type=int, default=1000)
    parser.add_argument(
        "--telemetry-points", type=int, default=3600,
        help="maximum live CPU/RAM/GPU samples retained in report JSON",
    )
    parser.add_argument(
        "--telemetry-seconds", type=float, default=2.0,
        help="CPU/RAM/GPU sampling interval (default: 2 seconds)",
    )
    parser.add_argument(
        "--watch-seconds", type=float, default=2.0,
        help="asynchronous refresh interval (default: 2 seconds; zero runs once)",
    )
    parser.add_argument(
        "--parent-pid", type=int,
        help="stop automatically when this training process exits",
    )
    args = parser.parse_args()
    if not args.log.is_file():
        parser.error(f"training log not found: {args.log}")
    if args.watch_seconds < 0:
        parser.error("--watch-seconds must be non-negative")
    if args.telemetry_points < 1:
        parser.error("--telemetry-points must be positive")
    if args.telemetry_seconds <= 0:
        parser.error("--telemetry-seconds must be positive")
    if args.watch_seconds > 0:
        try:
            asyncio.run(watch_report(args))
        except KeyboardInterrupt:
            print("report watcher stopped", flush=True)
        return
    else:
        monitor = SystemMonitor(args.parent_pid, max_points=args.telemetry_points)
        monitor.sample()
        report = build_report(args, telemetry=monitor.history)
        write_atomic(args.output, report)
        print(
            f"updated {args.output}: {len(report['training'])} training points, "
            f"{len(report['validation'])} validations",
            flush=True,
        )


if __name__ == "__main__":
    main()
