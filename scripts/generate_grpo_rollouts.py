"""Generate online GRPO response groups and score them with reward signals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
repository_root = str(Path(__file__).resolve().parents[1])
src_root = str(Path(__file__).resolve().parents[1] / "src")
sys.path[:] = [entry for entry in sys.path if str(Path(entry or ".").resolve()) not in {script_directory, repository_root, src_root}]
sys.path[:0] = [src_root, repository_root]

from inference.generator import Generator
from model.gpt import MiniGPT
from model.vocabulary import adapt_config_to_tokenizer, checkpoint_tokenizer_options
from post_training.online_grpo import OnlineRolloutGenerator, RolloutConfig, load_prompt_records, write_rollout_jsonl
from post_training.reward_model import RewardModel
from post_training.reward_scoring import RewardCalibration, RewardScorer
from tokenizer.encoder import Tokenizer
from training.checkpoint import load_checkpoint
from utils.config import load_yaml
from utils.device import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.gpu.yaml"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer-finetuning"))
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--reward-checkpoint", type=Path)
    parser.add_argument("--reward-calibration", type=Path)
    parser.add_argument("--prompt-file", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failure-output", type=Path)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reward-model-weight", type=float, default=1.0)
    parser.add_argument("--exact-match-weight", type=float, default=0.0)
    parser.add_argument("--normalize-reward-model", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reward-device", default="cpu")
    args = parser.parse_args()

    tokenizer = Tokenizer.load(args.tokenizer)
    config = adapt_config_to_tokenizer(load_yaml(args.model_config), tokenizer)
    device = resolve_device(args.device)
    policy = MiniGPT.from_config(config, device=device)
    load_checkpoint(args.policy_checkpoint, policy, map_location=device, use_ema=True, restore_rng=False, **checkpoint_tokenizer_options(tokenizer))
    generator = Generator(policy, tokenizer, device=device)

    scorer = None
    if args.reward_model_weight != 0:
        if args.reward_checkpoint is None:
            parser.error("--reward-checkpoint is required when --reward-model-weight is non-zero")
        reward_device = resolve_device(args.reward_device)
        reward_model = RewardModel(MiniGPT.from_config(config, device=reward_device), int(config["hidden_size"])).to(reward_device)
        load_checkpoint(args.reward_checkpoint, reward_model, map_location=reward_device, restore_rng=False, **checkpoint_tokenizer_options(tokenizer))
        calibration = RewardCalibration.load(args.reward_calibration) if args.reward_calibration else None
        if args.normalize_reward_model and calibration is None:
            parser.error("--normalize-reward-model requires --reward-calibration")
        scorer = RewardScorer(reward_model, tokenizer, max_length=args.max_sequence_length, calibration=calibration)

    rollout = OnlineRolloutGenerator(generator, scorer, RolloutConfig(
        group_size=args.group_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        reward_model_weight=args.reward_model_weight,
        exact_match_weight=args.exact_match_weight,
        normalize_reward_model=args.normalize_reward_model,
        seed=args.seed,
    ))
    groups, failures = rollout.generate(load_prompt_records(args.prompt_file), max_groups=args.max_groups)
    if not groups:
        parser.error("rollout generation produced no usable non-constant reward groups")
    write_rollout_jsonl(args.output, groups)
    if args.failure_output:
        write_rollout_jsonl(args.failure_output, failures)
    print(json.dumps({
        "output": str(args.output),
        "groups": len(groups),
        "failures": len(failures),
        "group_size": args.group_size,
    }, indent=2))


if __name__ == "__main__":
    main()
