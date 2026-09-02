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


def test_future_models_match_the_from_scratch_tokenizer() -> None:
    tokenizer = load_yaml(CONFIGS / "text/tokenizer.future.50k.yaml")
    pretraining = load_yaml(CONFIGS / "text/pretraining.future.fsdp.yaml")

    assert tokenizer["vocab_size"] == 50_000
    assert tokenizer["max_training_bytes"] == 4 * 1024**3
    assert tokenizer["source_sampling"] == "balanced_bytes"
    assert tokenizer["output_dir"] == "data/tokenizer-future-50k"
    for path in (CONFIGS / "text").glob("model.future.*.yaml"):
        assert load_yaml(path)["vocab_size"] == tokenizer["vocab_size"], path
    assert pretraining["train_files"] == [
        "data/shards/pretraining-future-50k/manifest.json"
    ]


def test_gpu_finetuning_profile_targets_balanced_quality() -> None:
    config = load_yaml(CONFIGS / "finetuning.gpu.yaml")
    weights = config["dataset_weights"]

    assert sum(weights.values()) == 1.0
    assert config["mixed_precision"] == "bf16"
    assert "grad_scaler_initial_scale" not in config
    assert "grad_scaler_growth_interval" not in config
    assert config["max_sequence_length"] == 384
    assert config["label_smoothing"] == 0.0
    assert weights["gsm8k"] >= 0.12
    assert sum(weights[name] for name in (
        "multilingual_bn_hi", "bangla_qa", "bangla_reading_qa", "v3_bengali_news",
    )) >= 0.14


def test_inference_defaults_to_finetuned_model_and_matching_tokenizer() -> None:
    config = load_yaml(CONFIGS / "inference.yaml")["serving"]

    assert config["checkpoint_path"] == "checkpoints/finetuning/best.pt"
    assert config["tokenizer_path"] == "data/tokenizer-v3"


def test_gpu_finetuning_includes_balanced_domain_expansion() -> None:
    config = load_yaml(CONFIGS / "finetuning.gpu.yaml")

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
    assert config["validation_metric_name"] == "dataset_weighted_v3_sft_domains_v1"


def test_expanded_sft_is_the_active_gpu_stage() -> None:
    config = load_yaml(CONFIGS / "finetuning.gpu.yaml")
    tokenizer = load_yaml(CONFIGS / "tokenizer.v2.yaml")

    assert config["epochs"] == 2
    assert config["learning_rate"] == pytest.approx(1e-5)
    assert config["samples_per_epoch"] == 300_000
    assert config["validation_batch_size"] > config["batch_size"]
    assert config["pad_to_multiple_of"] == 8
    assert sum(config["dataset_weights"].values()) == pytest.approx(1.0)
    assert config["validation_metric_name"] == "dataset_weighted_v3_sft_domains_v1"
    assert tokenizer["vocab_size"] == 38_000
    assert tokenizer["extension"]["base_tokenizer"] == "data/tokenizer-v2"
    assert tokenizer["extension"]["output_dir"] == "data/tokenizer-v3"


def test_expanded_finetuning_includes_every_tokenizer_source() -> None:
    tokenizer_sources = set(load_yaml(CONFIGS / "tokenizer.v2.yaml")["extension"]["sources"])
    config = load_yaml(CONFIGS / "finetuning.gpu.yaml")

    assert set(config["train_files"]) == tokenizer_sources
    assert "data/processed/preferences/validation.jsonl" in config["validation_files"]
    assert "data/processed/preferences/validation.jsonl" in config["validation_domains"]["english"]


def test_active_sft_fits_the_laptop_growth_route() -> None:
    config = load_yaml(CONFIGS / "finetuning.gpu.yaml")
    model = load_yaml(CONFIGS / "model.gpu.yaml")

    assert config["batch_size"] == 2
    assert config["gradient_accumulation_steps"] == 16
    assert config["learning_rate"] == pytest.approx(1e-5)
    assert config["epochs"] == 2
    assert config["validation_metric_name"] == "dataset_weighted_v3_sft_domains_v1"
    assert model["vocab_size"] == 38_000
    assert model["layers"] == 16


def test_dataset_catalog_lists_every_v3_training_source() -> None:
    config = load_yaml(CONFIGS / "finetuning.gpu.yaml")
    catalog = (ROOT / "docs" / "DATASET_CATALOG.md").read_text(encoding="utf-8")

    for name in config["dataset_weights"]:
        assert f"`{name}`" in catalog

    for manifest in (ROOT / "data" / "processed").glob("*/dataset-manifest.yaml"):
        assert f"`{manifest.parent.name}`" in catalog
