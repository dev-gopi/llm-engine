from pathlib import Path

from model.config import estimate_model_size
from utils.config import load_yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def test_all_model_and_training_profiles_are_complete() -> None:
    required_training = {
        "distributed_strategy",
        "distributed_backend",
        "checkpoint_format",
        "fused_optimizer",
        "evaluate_every",
        "mixed_precision",
        "gradient_accumulation_steps",
        "gradient_clip_norm",
    }
    for path in CONFIGS.glob("*.yaml"):
        config = load_yaml(path)
        if path.name.startswith("model."):
            assert estimate_model_size(config).parameters > 0
        if path.name.startswith(("pretraining", "finetuning", "training.")):
            assert required_training <= config.keys(), path
            assert config["distributed_strategy"] in {"ddp", "fsdp", "fsdp_hybrid", "none"}
            assert config["checkpoint_format"] in {"single_file", "distributed"}
