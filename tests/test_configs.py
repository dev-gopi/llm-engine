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


def test_v2_gpu_finetuning_profile_targets_balanced_quality() -> None:
    config = load_yaml(CONFIGS / "finetuning.v2.gpu.yaml")
    weights = config["dataset_weights"]

    assert sum(weights.values()) == 1.0
    assert config["mixed_precision"] == "fp16"
    assert config["grad_scaler_initial_scale"] == 1024
    assert config["grad_scaler_growth_interval"] == 20000
    assert config["max_sequence_length"] == 384
    assert config["label_smoothing"] == 0.0
    assert weights["gsm8k"] >= 0.12
    assert sum(weights[name] for name in (
        "multilingual_bn_hi", "bangla_qa", "bangla_reading_qa",
    )) >= 0.14
