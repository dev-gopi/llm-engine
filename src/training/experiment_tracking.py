"""First-class experiment tracking adapters with optional MLflow/W&B/TensorBoard imports."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

class Tracker:
    def log(self, metrics: dict[str, float], step: int | None = None) -> None: pass
    def close(self) -> None: pass

class NullTracker(Tracker):
    def log(self, metrics, step=None): pass

class TensorBoardTracker(Tracker):
    def __init__(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter
        self.writer=SummaryWriter(log_dir)
    def log(self, metrics, step=None):
        step=0 if step is None else step
        for k,v in metrics.items(): self.writer.add_scalar(k, float(v), step)
    def close(self): self.writer.close()

class MLflowTracker(Tracker):
    def __init__(self, experiment: str, run_name: str | None = None):
        import mlflow
        self.mlflow=mlflow; mlflow.set_experiment(experiment); self.run=mlflow.start_run(run_name=run_name)
    def log(self, metrics, step=None): self.mlflow.log_metrics({k:float(v) for k,v in metrics.items()}, step=step)
    def close(self): self.mlflow.end_run()

class WandBTracker(Tracker):
    def __init__(self, project: str, run_name: str | None = None, **kwargs):
        import wandb
        self.wandb=wandb; self.run=wandb.init(project=project, name=run_name, **kwargs)
    def log(self, metrics, step=None): self.wandb.log({k:float(v) for k,v in metrics.items()}, step=step)
    def close(self): self.wandb.finish()

def create_tracker(kind: str, **kwargs) -> Tracker:
    kind=kind.lower()
    if kind in {"none","null"}: return NullTracker()
    if kind in {"tensorboard","tb"}: return TensorBoardTracker(kwargs["log_dir"])
    if kind == "mlflow": return MLflowTracker(kwargs["experiment"], kwargs.get("run_name"))
    if kind in {"wandb","weights_and_biases"}: return WandBTracker(kwargs["project"], kwargs.get("run_name"), **kwargs.get("init_kwargs", {}))
    raise ValueError(f"unsupported tracker {kind!r}")
