"""Train or resume the Gopi causal language model."""

from __future__ import annotations

try:
    from scripts._bootstrap import PROJECT_ROOT  # noqa: F401
except ModuleNotFoundError:
    from _bootstrap import PROJECT_ROOT  # noqa: F401

import argparse
import atexit
from collections.abc import Mapping
from contextlib import nullcontext
import json
import hashlib
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

import torch

from model.gpt import MiniGPT
from datasets.governance import enforce_dataset_governance
from model.loss import CausalLanguageModelLoss
from optim.adamw import adamw_from_config
from optim.ema import EMA
from optim.scheduler import Scheduler
from tokenizer.encoder import Tokenizer
from training.generation_checkpoint import retention_passes, save_best_generation
from training.checkpoint import load_checkpoint, save_checkpoint
from training.distributed_checkpoint import load_distributed_checkpoint, save_distributed_checkpoint
from training.data import build_loader, interleave_loaders, _mixture_groups
from training.distributed import DistributedTrainer
from training.evaluator import Evaluator
from training.trainer import Trainer
from training.planner import optimizer_steps_for_epochs
from training.peft import LoRALinear, apply_lora, has_lora
from training.elastic import PreemptionCoordinator
from training.reporting import archive_previous_report_files
from evaluation.benchmarks import BenchmarkCase, score_answer, summarize_scores
from inference.context import format_system_prompt
from inference.generator import Generator
from dotenv import load_dotenv

from utils.config import apply_cli_defaults, load_yaml
from utils.device import verify_cuda_health
from utils.logger import configure_logging, get_logger
from utils.seed import set_seed

logger = get_logger(__name__)


def _evaluate_at_stage_start(
    enabled: bool, *, init_from: Path | None, resume: Path | None
) -> bool:
    """Run step-zero gates only when initializing a new training stage."""
    return bool(enabled and init_from is not None and resume is None)


def _stop_reporter(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def _start_reporter(args: argparse.Namespace, config: dict) -> subprocess.Popen | None:
    if args.no_live_report or int(os.getenv("RANK", "0")) != 0:
        return None
    command = [
        sys.executable,
        str(Path(__file__).with_name("build_training_report.py")),
        "--log", str(args.log_file),
        "--output", str(args.report_json),
        "--model-config", str(args.model_config),
        "--training-config", str(args.training_config),
        "--latest-checkpoint", str(args.output),
        "--best-checkpoint", str(args.best_output),
        "--watch-seconds", str(args.report_refresh_seconds),
        "--telemetry-seconds", str(args.report_telemetry_seconds),
        "--telemetry-points", str(args.report_telemetry_points),
        "--parent-pid", str(os.getpid()),
    ]
    generation_config = config.get("generation_evaluation") or {}
    if generation_config.get("enabled", False) and generation_config.get("output"):
        command.extend(["--generation-evaluation", str(generation_config["output"])])
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as error:
        logger.warning("Could not start standalone live reporter: %s", error)
        return None
    atexit.register(_stop_reporter, process)
    logger.info(
        "Started standalone live reporter pid=%d json=%s",
        process.pid, args.report_json,
    )
    return process


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.gpu.yaml"))
    parser.add_argument("--training-config", type=Path, default=Path("configs/pretraining.gpu.yaml"))
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--best-output", type=Path, default=None)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-from", type=Path, help="load model weights only for a new training stage")
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--log-file", type=Path, default=None,
        help="append training logs for the standalone live report viewer",
    )
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--report-refresh-seconds", type=float, default=None)
    parser.add_argument("--report-telemetry-seconds", type=float, default=None)
    parser.add_argument("--report-telemetry-points", type=int, default=None)
    parser.add_argument(
        "--no-live-report", action="store_true",
        help="do not launch the isolated JSON report watcher",
    )
    args = parser.parse_args()
    config = load_yaml(args.training_config)
    generation_config = config.get("generation_evaluation") or {}
    if not isinstance(generation_config, Mapping):
        parser.error("generation_evaluation must be a mapping")
    if config.get("planning_only", False):
        parser.error("planning-only training profile; use scripts/plan_training.py")
    model_config = load_yaml(args.model_config)
    if model_config.get("planning_only", False):
        parser.error("planning-only model profile; use a validated model configuration")
    apply_cli_defaults(args, config.get("runtime", {}), {
        "tokenizer": Path("data/tokenizer"),
        "output": Path("checkpoints/training/latest.pt"),
        "best_output": Path("checkpoints/training/best.pt"),
        "log_file": Path("logs/training.log"),
        "report_json": Path("reports/training_report.json"),
        "report_refresh_seconds": 2.0,
        "report_telemetry_seconds": 2.0,
        "report_telemetry_points": 3600,
    })
    if args.resume and args.init_from:
        parser.error("--resume and --init-from cannot be used together")
    required_init_checkpoint = config.get("required_init_checkpoint")
    if not args.init_from and not args.resume and required_init_checkpoint:
        args.init_from = Path(required_init_checkpoint)
    if config.get("require_init_from", False) and not (args.init_from or args.resume):
        parser.error("this post-training profile requires --init-from a completed checkpoint (or --resume its own interrupted run)")
    if required_init_checkpoint and args.init_from:
        if args.init_from.resolve() != Path(required_init_checkpoint).resolve():
            parser.error(
                f"this recovery profile requires --init-from {required_init_checkpoint}"
            )
    if config.get("require_prepared_data", False) and not config.get("prepared_data"):
        parser.error("prepare complete, decontaminated SFT records with scripts/prepare_sft_stage.py, then use its generated training.yaml")
    if args.init_from and args.init_from.resolve() in {args.output.resolve(), args.best_output.resolve()}:
        parser.error("new-stage output paths must differ from --init-from; preserve the completed checkpoint")
    # Fail before archiving logs or starting the report watcher.
    if config.get("require_cuda", False) and not torch.cuda.is_available():
        parser.error("this training profile requires CUDA; the completed checkpoint can still be evaluated on CPU")
    if config.get("require_cuda", False):
        try:
            verify_cuda_health()
        except RuntimeError as error:
            parser.error(str(error))
    try:
        _mixture_groups(config.get("train_files", []), [1] * len(config.get("train_files", [])), config)
    except ValueError as error:
        parser.error(str(error))

    generation_output_path = (
        Path(generation_config["output"])
        if generation_config.get("enabled", False) and generation_config.get("output")
        else None
    )
    if (
        generation_output_path is not None
        and generation_output_path.resolve() == args.report_json.resolve()
    ):
        parser.error(
            "generation_evaluation.output must differ from runtime.report_json; "
            "otherwise the live report recursively embeds itself"
        )
    archived_reports = archive_previous_report_files(
        args.log_file, args.report_json, resume=bool(args.resume),
        extra_paths=([generation_output_path] if generation_output_path else []),
    )
    configure_logging(log_file=args.log_file)
    for source, destination in archived_reports:
        logger.info("Archived previous report file %s to %s", source, destination)
    if args.resume:
        logger.info("Appending resumed training report data to %s", args.log_file)
    else:
        logger.info("Starting a fresh training report in %s", args.log_file)
    if args.report_refresh_seconds <= 0:
        parser.error("--report-refresh-seconds must be positive")
    if args.report_telemetry_seconds <= 0:
        parser.error("--report-telemetry-seconds must be positive")
    if args.report_telemetry_points < 1:
        parser.error("--report-telemetry-points must be positive")
    _start_reporter(args, config)
    precision = str(config.get("mixed_precision", "none"))
    if precision == "fp16" and not torch.cuda.is_available():
        parser.error(
            "the selected GPU training profile requires CUDA, but PyTorch cannot access a GPU. "
            "Fix the NVIDIA driver until `nvidia-smi` works, or explicitly select "
            "--model-config configs/model.cpu.yaml --training-config configs/pretraining.cpu.yaml"
        )
    if precision == "bf16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        parser.error(
            "the selected profile requires CUDA BF16, but this GPU does not support it; "
            "use an FP16 profile or set mixed_precision: fp16"
        )
    base_seed = int(config.get("seed", 42))
    set_seed(base_seed)
    governed_paths = [*config.get("train_files", []), *config.get("validation_files", [])]
    governance_findings = enforce_dataset_governance(
        governed_paths, config.get("dataset_governance"),
    )
    for finding in governance_findings:
        logger.warning("dataset governance [%s]: %s", finding.code, finding.message)
    distributed = DistributedTrainer.initialize(config.get("distributed_backend"))
    atexit.register(DistributedTrainer.shutdown)
    tokenizer = Tokenizer.load(args.tokenizer)
    if config.get("prepared_data") and config["prepared_data"].get("tokenizer_fingerprint") != tokenizer.fingerprint:
        parser.error("prepared dataset tokenizer changed; prepare the data again with the selected tokenizer")
    configured_vocab_size = int(model_config["vocab_size"])
    if tokenizer.vocab_size != configured_vocab_size:
        if (
            tokenizer.base_vocab_size == configured_vocab_size
            and tokenizer.vocab_size > configured_vocab_size
        ):
            model_config = dict(model_config)
            model_config["vocab_size"] = tokenizer.vocab_size
            logger.info(
                "Using append-only tokenizer extension: resizing vocabulary from %d to %d",
                configured_vocab_size, tokenizer.vocab_size,
            )
        else:
            parser.error(
                "tokenizer vocabulary does not match model vocab_size and is not a "
                "verified append-only extension of that vocabulary"
            )
    max_seq_len = int(config.get("max_sequence_length", 0))
    max_pos = int(model_config.get("max_position", 0))
    if max_seq_len > max_pos:
        parser.error(
            f"training max_sequence_length ({max_seq_len}) exceeds model max_position ({max_pos}). "
            f"Use a matching training config (e.g. max_sequence_length <= {max_pos}) or a model config with max_position >= {max_seq_len}."
        )
    logger.info("Building model on %s", distributed.device)
    model = MiniGPT.from_config(model_config, device=distributed.device)
    logger.info("Model initialized")
    if args.init_from:
        logger.info("Loading initial model weights from %s", args.init_from)
        # A new training stage should start from the learned model itself. EMA
        # can lag badly during short stages and is recreated for this run below.
        init_from_weights = str(config.get("init_from_weights", "model")).lower()
        if init_from_weights not in {"model", "ema"}:
            parser.error("init_from_weights must be 'model' or 'ema'")
        initial_state = load_checkpoint(
            args.init_from,
            model,
            map_location="cpu",
            use_ema=init_from_weights == "ema",
            restore_rng=False,
            expected_tokenizer_fingerprint=tokenizer.fingerprint,
            compatible_tokenizer_fingerprints=tokenizer.compatible_base_fingerprints,
            allow_vocab_extension=bool(tokenizer.compatible_base_fingerprints),
        )
        if init_from_weights == "ema" and not initial_state.get("ema_applied"):
            parser.error("init_from_weights='ema' requires an EMA payload in the initial checkpoint")
        initial_stage = initial_state.get("metadata", {}).get("training_stage_id")
        forbidden_initial_stages = {
            str(value) for value in config.get("forbidden_init_training_stage_ids", [])
        }
        if initial_stage in forbidden_initial_stages:
            parser.error(
                f"initial checkpoint belongs to rejected stage {initial_stage!r}; "
                "start this recovery run from the pre-SFT checkpoint"
            )
    peft_metadata = None
    if config.get("peft"):
        if has_lora(model):
            peft_metadata = dict(config["peft"])
            peft_metadata["matched_modules"] = [
                name for name, module in model.named_modules() if isinstance(module, LoRALinear)
            ]
            peft_metadata["trainable_parameters"] = sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            )
            peft_metadata["total_parameters"] = sum(
                parameter.numel() for parameter in model.parameters()
            )
        else:
            peft_metadata = apply_lora(model, config["peft"])
        logger.info(
            "Enabled LoRA PEFT: rank=%d matched=%d trainable=%d/%d (%.3f%%)",
            peft_metadata["rank"], len(peft_metadata["matched_modules"]),
            peft_metadata["trainable_parameters"], peft_metadata["total_parameters"],
            100.0 * peft_metadata["trainable_parameters"] / peft_metadata["total_parameters"],
        )
    strategy = str(config.get("distributed_strategy", "ddp"))
    distributed_checkpoints = strategy.startswith("fsdp") or str(
        config.get("checkpoint_format", "single_file")
    ).lower() == "distributed"
    training_model = DistributedTrainer.wrap(
        model,
        distributed,
        strategy=strategy,
        mixed_precision=str(config.get("mixed_precision", "none")),
    )
    # Model initialization must be identical across ranks, but stochastic
    # training operations (for example dropout) should not reuse identical RNG
    # streams on every worker.
    if distributed.world_size > 1 and not args.resume:
        set_seed(base_seed + distributed.rank)
    logger.info("Preparing training datasets")
    train_loader = build_loader(config["train_files"], tokenizer, config, shuffle=True, rank=distributed.rank, world_size=distributed.world_size)
    logger.info("Training loader ready: %d batches per epoch", len(train_loader))
    epochs = args.epochs or int(config.get("epochs", 1))
    accumulation = int(config.get("gradient_accumulation_steps", 1))
    total_steps = optimizer_steps_for_epochs(len(train_loader), epochs, accumulation)
    optimizer = adamw_from_config(training_model, config)
    scheduler = Scheduler.from_config(optimizer, config, total_steps=total_steps)
    # A conventional EMA duplicates every parameter and defeats FSDP memory
    # sharding. Large FSDP jobs should average selected exported checkpoints.
    ema_decay = config.get("ema_decay", 0.999)
    ema = None if strategy.startswith("fsdp") or ema_decay is None else EMA(
        training_model, decay=float(ema_decay)
    )
    loss_fn = CausalLanguageModelLoss.from_config(config)
    trainer = Trainer(
        training_model, optimizer, loss_fn, scheduler=scheduler, ema=ema,
        gradient_clip_norm=config.get("gradient_clip_norm", 1.0), device=distributed.device,
        gradient_accumulation_steps=accumulation,
        mixed_precision=str(config.get("mixed_precision", "none")),
        grad_scaler_initial_scale=float(config.get("grad_scaler_initial_scale", 65536.0)),
        grad_scaler_growth_interval=int(config.get("grad_scaler_growth_interval", 2000)),
        reasoning_trace_policy=str(config.get("reasoning_trace_policy", "optional")),
        mtp_loss_weight=float(config.get("mtp_loss_weight", 0.0)),
        moe_aux_loss_weight=float(config.get("moe_aux_loss_weight", 0.0)),
    )
    preemption = PreemptionCoordinator()
    preemption.install()
    atexit.register(preemption.restore)
    if args.resume:
        if args.resume.is_dir():
            logger.info("Restoring distributed checkpoint from %s", args.resume)
            state = load_distributed_checkpoint(
                args.resume, training_model, optimizer,
                scheduler=scheduler, scaler=trainer.scaler,
            )
            saved_fingerprint = state.get("tokenizer_fingerprint")
            if saved_fingerprint and saved_fingerprint != tokenizer.fingerprint:
                parser.error(
                    "checkpoint tokenizer fingerprint does not match the selected tokenizer"
                )
        else:
            logger.info("Restoring checkpoint from %s", args.resume)
            state = load_checkpoint(
                args.resume, model, optimizer=optimizer, scheduler=scheduler,
                ema=ema, scaler=trainer.scaler, map_location=distributed.device,
                expected_tokenizer_fingerprint=tokenizer.fingerprint,
            )
        expected_stage = config.get("training_stage_id")
        saved_metadata = state.get("metadata", state)
        saved_stage = saved_metadata.get("training_stage_id")
        if expected_stage is not None and saved_stage != expected_stage:
            parser.error(
                "checkpoint belongs to a different training stage; start this changed "
                "dataset/evaluation mixture with --init-from instead of --resume"
            )
        trainer.global_step = state["step"]
        trainer.load_state_dict(state.get("trainer", {}))
        if distributed.world_size > 1 and not distributed_checkpoints:
            # A single-file DDP checkpoint contains rank-zero RNG only. Avoid
            # cloning that stream across all workers after resume.
            set_seed(base_seed + distributed.rank + trainer.global_step)
        # Checkpoints contain the scaler's old tuning. Keep the current run's
        # configured growth interval when resuming so stability changes apply.
        if trainer.scaler.is_enabled():
            trainer.scaler.set_growth_interval(
                int(config.get("grad_scaler_growth_interval", 2000))
            )
        sampler_state = state.get("sampler")
        if sampler_state and hasattr(train_loader.batch_sampler, "load_state_dict"):
            # The trainer tracks consumed batches; older sampler snapshots only
            # retained the offset from the start of the resumed run.
            sampler_state = dict(sampler_state)
            if "batch_in_epoch" in state.get("trainer", {}):
                sampler_state["start_batch"] = trainer.batch_in_epoch
                sampler_state["epoch"] = trainer.current_epoch
            if (
                "batch_size" not in sampler_state
                and trainer.batch_in_epoch > train_loader.batch_sampler.total_batches
            ):
                parser.error(
                    "this legacy checkpoint does not record its training batch size, and its "
                    f"saved position ({trainer.batch_in_epoch} batches) exceeds the current "
                    f"epoch ({train_loader.batch_sampler.total_batches} batches). Resume once "
                    "with the checkpoint's original batch_size to create a compatible checkpoint."
                )
            train_loader.batch_sampler.load_state_dict(sampler_state)
            restored_batch = train_loader.batch_sampler.start_batch
            if restored_batch != trainer.batch_in_epoch:
                logger.info(
                    "Converted resume position from batch %d to %d for batch_size=%d",
                    trainer.batch_in_epoch, restored_batch,
                    train_loader.batch_sampler.batch_size,
                )
                trainer.batch_in_epoch = restored_batch

    validation_loader = None
    validation_weights = None
    evaluator = None
    if config.get("validation_files"):
        logger.info("Preparing validation datasets")
        validation_config = dict(config)
        validation_config["seed"] = int(config.get("validation_seed", config.get("seed", 42)))
        validation_batch_size = int(config.get("validation_batch_size", config.get("batch_size", 32)))
        if validation_batch_size < 1:
            parser.error("validation_batch_size must be positive")
        validation_config["batch_size"] = validation_batch_size
        validation_domains = config.get("validation_domains")
        if validation_domains:
            if not isinstance(validation_domains, dict):
                parser.error("validation_domains must be a mapping")
            balance_sources = bool(config.get("validation_balance_sources", False))
            fixed_subset = bool(config.get("validation_fixed_subset", False))
            validation_loader = {}
            for domain, paths in validation_domains.items():
                source_groups = [[path] for path in paths] if balance_sources else [paths]
                validation_loader[str(domain)] = interleave_loaders(
                    build_loader(
                        group, tokenizer, validation_config, shuffle=False,
                        # A deterministic shuffle makes max_batches a stable,
                        # representative subset instead of always taking the
                        # first records in each source.
                        sampler_shuffle=balance_sources or fixed_subset,
                        rank=distributed.rank, world_size=distributed.world_size,
                    )
                    for group in source_groups
                )
            validation_weights = config.get("validation_weights")
            if not isinstance(validation_weights, dict):
                parser.error("validation_weights must be provided with validation_domains")
        else:
            validation_loader = build_loader(
                config["validation_files"], tokenizer, validation_config,
                shuffle=False,
                sampler_shuffle=bool(config.get("validation_fixed_subset", False)),
                rank=distributed.rank, world_size=distributed.world_size,
            )
        evaluator = Evaluator(
            training_model,
            loss_fn=loss_fn,
            device=distributed.device,
            mixed_precision=precision,
            ema=ema if config.get("validation_use_ema", False) else None,
        )

    generation_cases: list[BenchmarkCase] = []
    generation_output: Path | None = None
    if generation_config.get("enabled", False):
        if strategy.startswith("fsdp"):
            parser.error("generation_evaluation is not supported with FSDP training")
        cases_path = Path(generation_config.get("cases", "configs/evaluation.domains.jsonl"))
        if not cases_path.is_file():
            parser.error(f"generation evaluation cases not found: {cases_path}")
        with cases_path.open(encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                generation_cases.append(BenchmarkCase(
                    item["category"], item["prompt"], tuple(item["expected"]),
                    tuple(item.get("forbidden", ())), item.get("match", "contains"),
                    item.get("max_answer_tokens"),
                ))
        if not generation_cases:
            parser.error("generation evaluation cases file is empty")
        if len({(case.category, case.prompt) for case in generation_cases}) != len(generation_cases):
            parser.error("generation evaluation cases must be unique")
        generation_output = Path(generation_config.get("output", "reports/generation_quality.json"))
        if generation_config.get("best_output"):
            generation_best = Path(generation_config["best_output"]).resolve()
            protected_paths = {Path(args.output).resolve(), Path(args.best_output).resolve()}
            protected_paths.update(path.resolve() for path in (args.init_from, args.resume) if path)
            if generation_best in protected_paths:
                parser.error("generation_evaluation.best_output must differ from training checkpoint paths")
        if int(generation_config.get("max_tokens", 64)) < 1:
            parser.error("generation_evaluation.max_tokens must be positive")
        if float(generation_config.get("repetition_penalty", 1.05)) <= 0:
            parser.error("generation_evaluation.repetition_penalty must be positive")
        if int(generation_config.get("no_repeat_ngram_size", 0)) < 0:
            parser.error("generation_evaluation.no_repeat_ngram_size must be non-negative")
        if generation_config.get("weights", "ema") not in {"ema", "model"}:
            parser.error("generation_evaluation.weights must be ema or model")
        if generation_config.get("preserve_passed", False) and not generation_config.get("best_output"):
            parser.error("generation_evaluation.preserve_passed requires best_output")

    last_generation_step: int | None = None
    last_generation_summary: dict[str, float | int] | None = None

    def validation_generation_callback(current: Trainer, epoch: int, _metrics, _domains):
        nonlocal last_generation_step, last_generation_summary
        if not generation_cases:
            return None
        if current.global_step == last_generation_step:
            return last_generation_summary
        last_generation_step = current.global_step
        DistributedTrainer.barrier(distributed)
        summary = None
        if distributed.is_main_process:
            was_training = model.training
            started = time.perf_counter()
            try:
                from datasets.preprocessor import format_messages

                inference_config = load_yaml(generation_config.get("inference_config", "configs/inference.yaml"))
                system_prompt = format_system_prompt(
                    str(inference_config.get("system_prompt", "You are Gopi, a helpful assistant.")),
                    str(inference_config.get("response_format", "plain")),
                    include_safety_instruction=bool(inference_config.get("embed_safety_instruction", True)),
                )
                use_generation_ema = ema is not None and generation_config.get("weights", "ema") == "ema"
                parameter_context = (
                    ema.average_parameters(training_model, backup_device="cpu")
                    if use_generation_ema else nullcontext()
                )
                with parameter_context:
                    generator = Generator(model, tokenizer, device=distributed.device)
                    scored = []
                    results = []
                    for case in generation_cases:
                        prompt = format_messages([
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": case.prompt},
                        ], add_generation_prompt=True)
                        generated = generator.generate(
                            prompt,
                            max_tokens=int(generation_config.get("max_tokens", 64)),
                            temperature=0.0,
                            top_k=0,
                            repetition_penalty=float(generation_config.get("repetition_penalty", 1.05)),
                            no_repeat_ngram_size=int(generation_config.get("no_repeat_ngram_size", 0)),
                            allow_special_tokens=True,
                        )
                        score = score_answer(generated.text, case)
                        scored.append((case, score))
                        results.append({
                            "category": case.category, "prompt": case.prompt,
                            "answer": generated.text, "score": score,
                        })
                    summary = summarize_scores(scored)
                    if generation_config.get("best_output"):
                        signature = hashlib.sha256(json.dumps({
                            "cases": cases_path.read_text(encoding="utf-8"),
                            "system_prompt": system_prompt,
                            "max_tokens": int(generation_config.get("max_tokens", 64)),
                            "repetition_penalty": float(generation_config.get("repetition_penalty", 1.05)),
                            "no_repeat_ngram_size": int(generation_config.get("no_repeat_ngram_size", 0)),
                            "tokenizer": tokenizer.fingerprint,
                            "ema_used": use_generation_ema,
                            "scorer_version": 2,
                            "preserve_passed": bool(generation_config.get("preserve_passed", False)),
                        }, sort_keys=True).encode()).hexdigest()
                        case_scores = {
                            json.dumps([case.category, case.prompt], ensure_ascii=False): score
                            for case, score in scored
                        }
                        retention_ok = (
                            retention_passes(
                                generation_config["best_output"],
                                evaluation_signature=signature,
                                case_scores=case_scores,
                            )
                            if generation_config.get("preserve_passed", False) else True
                        )
                        summary["retention_passed"] = retention_ok
                        if save_best_generation(
                            generation_config["best_output"], model,
                            accuracy=summary["accuracy"], evaluation_signature=signature,
                            step=current.global_step,
                            case_scores=case_scores,
                            preserve_passed=bool(generation_config.get("preserve_passed", False)),
                            metadata={"model_config": model_config,
                                      "tokenizer_fingerprint": tokenizer.fingerprint,
                                      "ema_used": use_generation_ema,
                                      "peft": peft_metadata},
                        ):
                            logger.info("new_best_generation step=%d accuracy=%.4f checkpoint=%s",
                                        current.global_step, summary["accuracy"],
                                        generation_config["best_output"])

                report = {
                    "checkpoint": str(args.output), "step": current.global_step,
                    "epoch": epoch + 1, "ema_used": use_generation_ema,
                    "summary": summary, "results": results,
                }
                generation_output.parent.mkdir(parents=True, exist_ok=True)
                descriptor, temporary = tempfile.mkstemp(
                    prefix=f".{generation_output.name}.", dir=generation_output.parent
                )
                try:
                    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                        json.dump(report, stream, indent=2, ensure_ascii=False)
                        stream.write("\n")
                    os.replace(temporary, generation_output)
                except BaseException:
                    Path(temporary).unlink(missing_ok=True)
                    raise
                # Retain answers from every check instead of losing the evidence
                # when the next validation replaces the live report.
                history_dir = generation_output.parent / (generation_output.stem + "_history")
                history_dir.mkdir(parents=True, exist_ok=True)
                with (history_dir / f"step-{current.global_step}-{time.time_ns()}.json").open(
                    "w", encoding="utf-8"
                ) as stream:
                    json.dump(report, stream, indent=2, ensure_ascii=False)
                    stream.write("\n")
                category_metrics = " ".join(
                    f"{key.removeprefix('accuracy_')}={value:.4f}"
                    for key, value in summary.items() if key.startswith("accuracy_")
                )
                logger.info(
                    "generation_evaluation epoch=%d step=%d accuracy=%.4f cases=%d %s duration_seconds=%.2f",
                    epoch + 1, current.global_step, float(summary["accuracy"]),
                    int(summary["cases"]), category_metrics,
                    time.perf_counter() - started,
                )
                last_generation_summary = summary
            except Exception:
                if generation_config.get("preserve_passed", False):
                    # A required retention check must not silently fail open.
                    # All ranks receive this failure after the barrier below.
                    summary = {"retention_evaluation_failed": True}
                logger.exception("generation evaluation failed at step=%d; required=%s", current.global_step,
                                 bool(generation_config.get("preserve_passed", False)))
            finally:
                model.train(was_training)
        DistributedTrainer.barrier(distributed)
        if generation_config.get("preserve_passed", False):
            failed = torch.tensor(int(bool(summary and summary.get("retention_evaluation_failed"))),
                                  device=distributed.device)
            if distributed.world_size > 1:
                torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX)
            if failed.item():
                raise RuntimeError("required retention evaluation failed; training stopped")
        return summary

    def checkpoint_callback(current: Trainer, epoch: int) -> None:
        sampler_state = train_loader.batch_sampler.state_dict()
        sampler_state["start_batch"] = current.batch_in_epoch
        if distributed_checkpoints:
            save_distributed_checkpoint(
                args.output,
                training_model,
                optimizer,
                scheduler=scheduler,
                scaler=current.scaler,
                metadata={
                    "step": current.global_step,
                    "epoch": epoch + 1,
                    "model_config": model_config,
                    "tokenizer_fingerprint": tokenizer.fingerprint,
                    "training_stage_id": config.get("training_stage_id"),
                    "peft": peft_metadata,
                    "trainer": current.state_dict(),
                    "sampler": sampler_state,
                },
            )
            return
        if not distributed.is_main_process:
            return
        save_checkpoint(args.output, model, optimizer=optimizer, scheduler=scheduler, ema=ema, scaler=current.scaler,
                        step=current.global_step, metadata={"epoch": epoch + 1, "model_config": model_config, "tokenizer_fingerprint": tokenizer.fingerprint,
                        "training_stage_id": config.get("training_stage_id"), "peft": peft_metadata},
                        trainer=current.state_dict(), sampler=sampler_state)

    def best_checkpoint_callback(current: Trainer, epoch: int) -> None:
        sampler_state = train_loader.batch_sampler.state_dict()
        sampler_state["start_batch"] = current.batch_in_epoch
        if distributed_checkpoints:
            save_distributed_checkpoint(
                args.best_output,
                training_model,
                optimizer,
                scheduler=scheduler,
                scaler=current.scaler,
                metadata={
                    "step": current.global_step,
                    "epoch": epoch + 1,
                    "validation_loss": current.best_validation_loss,
                    "best": True,
                    "model_config": model_config,
                    "tokenizer_fingerprint": tokenizer.fingerprint,
                    "training_stage_id": config.get("training_stage_id"),
                    "peft": peft_metadata,
                    "trainer": current.state_dict(),
                    "sampler": sampler_state,
                },
            )
            return
        if not distributed.is_main_process:
            return
        save_checkpoint(
            args.best_output, model, optimizer=optimizer, scheduler=scheduler, ema=ema,
            scaler=current.scaler, step=current.global_step,
            metadata={"epoch": epoch + 1, "validation_loss": current.best_validation_loss, "best": True, "model_config": model_config,
                      "tokenizer_fingerprint": tokenizer.fingerprint,
                      "training_stage_id": config.get("training_stage_id"), "peft": peft_metadata},
            trainer=current.state_dict(), sampler=sampler_state,
        )

    evaluate_generation_at_start = _evaluate_at_stage_start(
        bool(generation_config.get("evaluate_at_start", False)),
        init_from=args.init_from,
        resume=args.resume,
    )
    if evaluate_generation_at_start and generation_cases:
        validation_generation_callback(trainer, trainer.current_epoch - 1, {}, {})
        # Generation allocates many differently sized KV-cache and logits
        # blocks. Release its unused CUDA reservations before the full-context
        # step-zero validation pass, especially on 4 GiB devices.
        if distributed.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(distributed.device) / 1024**2
            reserved = torch.cuda.memory_reserved(distributed.device) / 1024**2
            torch.cuda.empty_cache()
            logger.info(
                "Released generation CUDA cache before validation: "
                "allocated_mb=%.1f reserved_mb=%.1f->%.1f",
                allocated, reserved,
                torch.cuda.memory_reserved(distributed.device) / 1024**2,
            )

    logger.info("Starting training at optimizer step %d; batch_size=%s accumulation=%d log_every=%s",
                trainer.global_step, config.get("batch_size"), accumulation, config.get("log_every", 10))
    validation_protocol = {
        "name": config.get("validation_metric_name"),
        "domains": config.get("validation_domains"),
        "weights": config.get("validation_weights"),
        "max_batches": config.get("validation_max_batches"),
        "seed": config.get("validation_seed", config.get("seed", 42)),
        "fixed_subset": config.get("validation_fixed_subset", False),
        "balance_sources": config.get("validation_balance_sources", False),
        "use_ema": config.get("validation_use_ema", False),
        "batch_size": config.get("validation_batch_size", config.get("batch_size")),
        "max_sequence_length": config.get("max_sequence_length"),
    }
    validation_protocol_id = hashlib.sha256(
        json.dumps(validation_protocol, sort_keys=True).encode()
    ).hexdigest()[:12]
    resolved_validation_metric_name = (
        f"{config.get('validation_metric_name', 'validation_loss')}:{validation_protocol_id}"
    )
    logger.info(
        "run_configuration=%s",
        json.dumps(
            {"model_config": model_config, "training_config": config},
            sort_keys=True, separators=(",", ":"),
        ),
    )
    evaluate_validation_at_start = _evaluate_at_stage_start(
        bool(config.get("validation_evaluate_at_start", False)),
        init_from=args.init_from,
        resume=args.resume,
    )
    # A run created before initial-best support can resume with domain
    # baselines but no best.pt. Re-evaluate its restored weights once and
    # establish a recoverable baseline instead of waiting forever for all
    # promotion gates to pass simultaneously.
    save_initial_best_checkpoint = bool(config.get("save_initial_best_checkpoint", False))
    evaluate_validation_at_start = (
        evaluate_validation_at_start or save_initial_best_checkpoint
    )
    history = trainer.fit(
        train_loader, epochs=epochs, evaluator=evaluator,
        validation_dataloader=validation_loader,
        validation_weights=validation_weights,
        validation_max_batches=config.get("validation_max_batches"),
        validation_progress_every=int(config.get("validation_progress_every", 0)),
        validation_evaluate_at_start=evaluate_validation_at_start,
        save_initial_best_checkpoint=save_initial_best_checkpoint,
        log_every=int(config.get("log_every", 10)),
        log_interval_seconds=(
            float(config["log_interval_seconds"])
            if config.get("log_interval_seconds") is not None else None
        ),
        evaluate_every=config.get("evaluate_every"),
        checkpoint_every=config.get("checkpoint_every"),
        checkpoint_callback=checkpoint_callback,
        best_checkpoint_callback=best_checkpoint_callback,
        early_stopping_patience=config.get("early_stopping_patience"),
        early_stopping_min_delta=float(config.get("early_stopping_min_delta", 0.0)),
        validation_lr_decay_factor=(
            float(config["validation_lr_decay_factor"])
            if config.get("validation_lr_decay_factor") is not None else None
        ),
        validation_lr_patience=int(config.get("validation_lr_patience", 1)),
        validation_lr_min_scale=float(config.get("validation_lr_min_scale", 0.1)),
        validation_lr_min_steps_between_decays=int(
            config.get("validation_lr_min_steps_between_decays", 0)
        ),
        validation_control_domain=config.get("validation_control_domain"),
        best_checkpoint_domain_max_regression=config.get(
            "best_checkpoint_domain_max_regression"
        ),
        best_checkpoint_min_generation_accuracy=(
            float(config["best_checkpoint_min_generation_accuracy"])
            if config.get("best_checkpoint_min_generation_accuracy") is not None else None
        ),
        validation_metric_name=resolved_validation_metric_name,
        validation_callback=validation_generation_callback if generation_cases else None,
        stop_requested=lambda: preemption.should_stop(distributed.device),
    )
    final_epoch = int(history[-1]["epoch"]) - 1 if history else trainer.current_epoch - 1
    checkpoint_callback(trainer, final_epoch)
    if distributed.is_main_process:
        print(json.dumps({
            "checkpoint": str(args.output), "best_checkpoint": str(args.best_output),
            "step": trainer.global_step, "stopped_early": trainer.stopped_early, "history": history,
        }, indent=2))
    DistributedTrainer.shutdown()


if __name__ == "__main__":
    main()
