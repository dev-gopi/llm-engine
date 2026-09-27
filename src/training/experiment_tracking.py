"""First-class experiment tracking adapters with optional MLflow/W&B/TensorBoard imports."""

from __future__ import annotations


class Tracker:
    def log(self, metrics: dict[str, float], step: int | None = None) -> None:
        pass

    def close(self) -> None:
        pass


class NullTracker(Tracker):
    def log(self, metrics, step=None):
        pass


class TensorBoardTracker(Tracker):
    def __init__(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir)

    def log(self, metrics, step=None):
        step = 0 if step is None else step
        for k, v in metrics.items():
            self.writer.add_scalar(k, float(v), step)

    def close(self):
        self.writer.close()


class MLflowTracker(Tracker):
    def __init__(self, experiment: str, run_name: str | None = None):
        import mlflow

        self.mlflow = mlflow
        mlflow.set_experiment(experiment)
        self.run = mlflow.start_run(run_name=run_name)

    def log(self, metrics, step=None):
        self.mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def close(self):
        self.mlflow.end_run()


class WandBTracker(Tracker):
    def __init__(self, project: str, run_name: str | None = None, **kwargs):
        import wandb

        self.wandb = wandb
        self.run = wandb.init(project=project, name=run_name, **kwargs)

    def log(self, metrics, step=None):
        self.wandb.log({k: float(v) for k, v in metrics.items()}, step=step)

    def close(self):
        self.wandb.finish()


def create_tracker(kind: str, **kwargs) -> Tracker:
    kind = kind.lower()
    if kind in {"none", "null"}:
        return NullTracker()
    if kind in {"tensorboard", "tb"}:
        return TensorBoardTracker(kwargs["log_dir"])
    if kind == "mlflow":
        return MLflowTracker(kwargs["experiment"], kwargs.get("run_name"))
    if kind in {"wandb", "weights_and_biases"}:
        return WandBTracker(
            kwargs["project"], kwargs.get("run_name"), **kwargs.get("init_kwargs", {})
        )
    raise ValueError(f"unsupported tracker {kind!r}")


def create_tracker_from_config(config, *, is_main_process: bool = True) -> Tracker:
    """Build a tracker from a training config on the main process only.

    Accepted shape::

        experiment_tracking:
          kind: tensorboard | mlflow | wandb | none
          ... backend-specific options ...
    """
    if not is_main_process:
        return NullTracker()
    tracking = config.get("experiment_tracking") or {"kind": "none"}
    if not isinstance(tracking, dict):
        raise ValueError("experiment_tracking must be a mapping")
    kwargs = {key: value for key, value in tracking.items() if key != "kind"}
    return create_tracker(str(tracking.get("kind", "none")), **kwargs)


def log_history(tracker: Tracker, history, *, prefix: str = "train") -> None:
    """Log numeric history rows without imposing a trainer-specific schema."""
    for index, row in enumerate(history or []):
        if not isinstance(row, dict):
            continue
        step = row.get("step")
        metrics = {
            f"{prefix}/{key}": float(value)
            for key, value in row.items()
            if key not in {"step", "epoch"}
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        }
        if metrics:
            tracker.log(metrics, step=int(step) if isinstance(step, int) else index + 1)
