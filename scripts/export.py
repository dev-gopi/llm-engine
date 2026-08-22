"""Export a trained Gopi model to deployment formats."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

import torch
from safetensors.torch import save_model

from model.gpt import MiniGPT
from training.checkpoint import load_checkpoint
from utils.config import load_yaml


def export_model(model: MiniGPT, output: Path, export_format: str, *, sequence_length: int = 16) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    example = torch.zeros((1, sequence_length), dtype=torch.long)
    if export_format == "safetensors":
        save_model(model, output, metadata={"format": "pt", "architecture": "MiniGPT"})
    elif export_format == "torch_export":
        exported = torch.export.export(model, (example,))
        torch.export.save(exported, output)
    elif export_format == "onnx":
        torch.onnx.export(
            model, example, output, input_names=["input_ids"], output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}, "logits": {0: "batch", 1: "sequence"}},
            opset_version=17,
        )
    else:
        raise ValueError("format must be safetensors, torch_export, or onnx")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latest/model.pt"))
    parser.add_argument("--format", choices=("safetensors", "torch_export", "onnx"), default="safetensors")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sequence-length", type=int, default=16)
    args = parser.parse_args()
    suffixes = {"safetensors": ".safetensors", "torch_export": ".pt2", "onnx": ".onnx"}
    output = args.output or Path("exports") / args.format / f"gopi{suffixes[args.format]}"
    config = load_yaml(args.model_config)
    model = MiniGPT.from_config(config, device="cpu")
    load_checkpoint(args.checkpoint, model, use_ema=True)
    artifact = export_model(model, output, args.format, sequence_length=args.sequence_length)
    destination_config = artifact.parent / "model.yaml"
    shutil.copy2(args.model_config, destination_config)
    if args.tokenizer.exists():
        destination_tokenizer = artifact.parent / "tokenizer"
        if destination_tokenizer.exists():
            shutil.rmtree(destination_tokenizer)
        shutil.copytree(args.tokenizer, destination_tokenizer)
    print(json.dumps({"artifact": str(artifact), "model_config": str(destination_config)}, indent=2))


if __name__ == "__main__":
    main()
