"""Export a trained Gopi model to deployment formats."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

import torch
import yaml
from safetensors.torch import save_file, save_model

from model.gpt import MiniGPT
from model.vocabulary import adapt_config_to_tokenizer, checkpoint_tokenizer_options
from tokenizer.encoder import Tokenizer
from training.checkpoint import load_checkpoint
from training.peft import merge_and_unload
from utils.config import load_yaml
from inference.quantization import quantize_int4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_export_manifest(
    artifact: Path, *, config: dict, tokenizer: Tokenizer, export_format: str,
    weight_dtype: str,
) -> Path:
    """Write a deterministic content-addressed deployment manifest.

    Paths are relative to the export directory so an immutable artifact bundle
    can be relocated without invalidating its compatibility metadata.
    """
    root = artifact.parent
    model_config = root / "model.yaml"
    tokenizer_root = root / "tokenizer"
    tokenizer_files = {
        file.relative_to(root).as_posix(): _sha256(file)
        for file in sorted(tokenizer_root.rglob("*")) if file.is_file()
    }
    manifest = {
        "schema_version": 1,
        "architecture": "MiniGPT",
        "export_format": export_format,
        "weight_dtype": weight_dtype,
        "artifacts": {artifact.name: _sha256(artifact)},
        "model_config": {"path": model_config.name, "sha256": _sha256(model_config), "values": config},
        "tokenizer": {"fingerprint": tokenizer.fingerprint, "files": tokenizer_files},
    }
    destination = root / "manifest.json"
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def _export_int4_safetensors(model: MiniGPT, output: Path) -> None:
    tensors: dict[str, torch.Tensor] = {}
    manifest: dict[str, dict[str, object]] = {}
    for name, value in model.state_dict().items():
        if value.is_floating_point():
            packed, scale, original_numel = quantize_int4(value)
            tensors[f"{name}.int4_packed"] = packed
            tensors[f"{name}.int4_scale"] = scale
            manifest[name] = {"shape": list(value.shape), "numel": original_numel}
        else:
            tensors[name] = value.detach().cpu().contiguous()
    save_file(
        tensors, output,
        metadata={"format": "llm-engine.int4.v1", "architecture": "MiniGPT", "int4_manifest": json.dumps(manifest)},
    )


def export_model(
    model: MiniGPT, output: Path, export_format: str, *, sequence_length: int = 16,
    weight_dtype: str = "float32",
) -> Path:
    """Export float32/float16/bfloat16 weights, or portable packed INT4 safetensors."""
    output.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    dtypes = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if weight_dtype not in {*dtypes, "int4"}:
        raise ValueError("weight_dtype must be float32, float16, bfloat16, or int4")
    if weight_dtype == "int4":
        if export_format != "safetensors":
            raise ValueError("INT4 export is supported only for safetensors")
        _export_int4_safetensors(model, output)
        return output
    model.to(dtype=dtypes[weight_dtype])
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
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.gpu.yaml"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/finetuning/best.pt"))
    parser.add_argument("--format", choices=("safetensors", "torch_export", "onnx"), default="safetensors")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument(
        "--weight-dtype", choices=("float32", "float16", "bfloat16", "int4"), default="float32",
        help="persist float weights at this dtype, or write a portable packed INT4 safetensors artifact",
    )
    parser.add_argument("--merge-lora", action="store_true", help="fold a loaded LoRA adapter into base weights")
    args = parser.parse_args()
    suffixes = {"safetensors": ".safetensors", "torch_export": ".pt2", "onnx": ".onnx"}
    output = args.output or Path("exports") / args.format / f"gopi{suffixes[args.format]}"
    if not args.tokenizer.exists():
        parser.error(f"tokenizer does not exist: {args.tokenizer}")
    tokenizer = Tokenizer.load(args.tokenizer)
    try:
        config = adapt_config_to_tokenizer(load_yaml(args.model_config), tokenizer)
    except ValueError as error:
        parser.error(str(error))
    model = MiniGPT.from_config(config, device="cpu")
    load_checkpoint(
        args.checkpoint, model, use_ema=True,
        **checkpoint_tokenizer_options(tokenizer),
    )
    if args.merge_lora:
        merge_and_unload(model)
    artifact = export_model(
        model, output, args.format, sequence_length=args.sequence_length,
        weight_dtype=args.weight_dtype,
    )
    destination_config = artifact.parent / "model.yaml"
    destination_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    destination_tokenizer = artifact.parent / "tokenizer"
    if destination_tokenizer.exists():
        shutil.rmtree(destination_tokenizer)
    shutil.copytree(args.tokenizer, destination_tokenizer)
    manifest = write_export_manifest(
        artifact, config=config, tokenizer=tokenizer, export_format=args.format,
        weight_dtype=args.weight_dtype,
    )
    print(json.dumps({
        "artifact": str(artifact), "model_config": str(destination_config),
        "manifest": str(manifest), "weight_dtype": args.weight_dtype,
    }, indent=2))


if __name__ == "__main__":
    main()
