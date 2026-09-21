"""Generate auditable model/dataset/reproducibility release artifacts."""
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import argparse
from pathlib import Path

from release.artifacts import DatasetCard, ModelCard, build_reproducibility_manifest


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/release")
    args=parser.parse_args()
    out=Path(args.output); out.mkdir(parents=True, exist_ok=True)
    model=ModelCard(
        model_name="Gopi LLM", version="0.1.0", architecture="decoder-only-transformer",
        intended_use="Local and controlled application inference.",
        limitations=("Evaluation fixtures are not a substitute for broad external benchmarks.", "Vision and audio are not advertised by the text serving endpoint."),
        training_data=("Configured training manifests",),
        evaluation_summary={}, safety_summary={}, provenance={"release_directory": str(out)},
    )
    dataset=DatasetCard(
        name="Gopi training corpus", version="0.1.0", sources=("Repository-configured datasets",),
        license_notes="Verify each source license before redistribution.",
        preprocessing=("Tokenizer normalization", "Repository-configured filtering and deduplication"),
        deduplication="Use the repository dataset governance and deduplication reports.",
        contamination_controls=("Held-out evaluation fixtures", "Dynamic fixture generation"),
        limitations=("Source-level licensing/provenance remains dataset-specific.",), hashes={},
    )
    manifest=build_reproducibility_manifest(root=".", config_paths=["configs/inference.yaml"])
    (out/"MODEL_CARD.md").write_text(model.to_markdown(), encoding="utf-8")
    (out/"DATASET_CARD.md").write_text(dataset.to_markdown(), encoding="utf-8")
    import json
    (out/"REPRODUCIBILITY.md").write_text("# Reproducibility\n\n```json\n"+json.dumps(manifest.to_dict(),indent=2,sort_keys=True)+"\n```\n",encoding="utf-8")
    return 0

if __name__ == "__main__": raise SystemExit(main())
