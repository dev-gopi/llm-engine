from pathlib import Path

import pytest

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
            for key in ("dataset_weights", "validation_weights"):
                if key in config:
                    assert sum(float(value) for value in config[key].values()) == pytest.approx(1.0), (
                        path, key
                    )


def test_v2_gpu_finetuning_profile_targets_balanced_quality() -> None:
    config = load_yaml(CONFIGS / "finetuning.v2.gpu.yaml")
    weights = config["dataset_weights"]

    assert sum(weights.values()) == 1.0
    assert config["mixed_precision"] == "bf16"
    assert "grad_scaler_initial_scale" not in config
    assert "grad_scaler_growth_interval" not in config
    assert config["max_sequence_length"] == 384
    assert config["label_smoothing"] == 0.0
    assert weights["gsm8k"] >= 0.12
    assert sum(weights[name] for name in (
        "multilingual_bn_hi", "bangla_qa", "bangla_reading_qa",
    )) >= 0.14


def test_v3_gpu_finetuning_includes_balanced_domain_expansion() -> None:
    config = load_yaml(CONFIGS / "finetuning.v3.gpu.yaml")

    expected_additions = {
        "v3_bengali_news": "bengali",
        "v3_hindi_news": "hindi",
        "v3_openassistant_en": "chat",
        "v3_code_feedback": "coding",
        "v3_math_instruct": "gsm8k",
    }
    for name, domain in expected_additions.items():
        train = f"data/processed/{name}/train.jsonl"
        validation = f"data/processed/{name}/validation.jsonl"
        assert config["dataset_weights"][name] > 0
        assert train in config["train_files"]
        assert validation in config["validation_files"]
        assert validation in config["validation_domains"][domain]

    assert "data/processed/hindi_hinglish/train.jsonl" in config["train_files"]
    assert "data/processed/hindi_hinglish/validation.jsonl" in config["validation_files"]
    assert "data/processed/hindi_hinglish/validation.jsonl" in config["validation_domains"]["hindi"]
    assert config["validation_weights"]["hindi"] == pytest.approx(0.12)
    assert sum(config["dataset_weights"].values()) == pytest.approx(1.0)
    assert sum(config["validation_weights"].values()) == pytest.approx(1.0)
    assert config["validation_metric_name"] == "dataset_weighted_v3_direct_sft_domains_v1"


def test_v2_expanded_sft_is_a_conservative_new_stage() -> None:
    config = load_yaml(CONFIGS / "finetuning.v2.expanded.gpu.yaml")
    tokenizer = load_yaml(CONFIGS / "tokenizer.v3.extension.yaml")

    assert config["epochs"] == 2
    assert config["learning_rate"] == pytest.approx(1e-5)
    assert config["samples_per_epoch"] == 300_000
    assert sum(config["dataset_weights"].values()) == pytest.approx(1.0)
    assert config["validation_metric_name"] == "dataset_weighted_v2_expanded_sft_domains_v1"
    assert tokenizer["vocab_size"] == 38_000
    assert tokenizer["extension"]["base_tokenizer"] == "data/tokenizer-v2-extended"
    assert tokenizer["extension"]["output_dir"] == "data/tokenizer-v3-extended-38k"


def test_v3_direct_sft_fits_the_laptop_growth_route() -> None:
    config = load_yaml(CONFIGS / "finetuning.v3.gpu.yaml")
    model = load_yaml(CONFIGS / "model.v3.gpu.yaml")

    assert config["batch_size"] == 1
    assert config["gradient_accumulation_steps"] == 32
    assert config["learning_rate"] == pytest.approx(1e-5)
    assert config["epochs"] == 2
    assert config["validation_metric_name"] == "dataset_weighted_v3_direct_sft_domains_v1"
    assert model["vocab_size"] == 38_000
    assert model["layers"] == 16
