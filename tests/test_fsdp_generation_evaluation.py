from pathlib import Path


def test_main_trainer_generation_evaluation_supports_fsdp_materialization() -> None:
    source = Path("scripts/train.py").read_text(encoding="utf8")
    assert "generation_evaluation is not supported with FSDP training" not in source
    assert "copy_fsdp_weights_to_model(training_model, generation_model)" in source
    assert "generation_model = MiniGPT.from_config" in source
    assert "Generator(generation_model" in source
    assert 'save_best_generation(\n                            generation_config["best_output"], generation_model,' in source
