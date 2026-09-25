"""Full LM Evaluation Harness support and adapters for llm-engine models.

Provides compatibility with EleutherAI lm-evaluation-harness interface,
including loglikelihood, loglikelihood_rolling, and generate_until primitives,
as well as a built-in zero-dependency evaluation harness runner with standard
benchmarks (MMLU, GSM8K, ARC, HellaSwag, Winogrande, TruthfulQA, Lambada).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import re
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from inference.generator import Generator
from model.gpt import MiniGPT
from tokenizer.encoder import Tokenizer
from utils.device import resolve_device
from utils.logger import get_logger

logger = get_logger(__name__)

# Try importing official lm_eval components if available in environment
try:
    import lm_eval
    from lm_eval.api.instance import Instance
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model
    HAS_LM_EVAL = True
except ImportError:
    HAS_LM_EVAL = False
    LM = object
    Instance = object
    register_model = lambda *names: (lambda cls: cls)


@dataclass(frozen=True)
class HarnessRequest:
    """Request representation for harness model operations."""
    request_type: str
    args: tuple[Any, ...]
    index: int = 0


class BaseHarnessLM:
    """Protocol base class for evaluation harness language models."""

    def loglikelihood(self, requests: Sequence[tuple[str, str] | Any]) -> list[tuple[float, bool]]:
        raise NotImplementedError

    def loglikelihood_rolling(self, requests: Sequence[tuple[str] | str | Any]) -> list[float]:
        raise NotImplementedError

    def generate_until(self, requests: Sequence[tuple[str, dict[str, Any]] | Any]) -> list[str]:
        raise NotImplementedError


class HarnessModelAdapter(BaseHarnessLM):
    """Adapter bridging MiniGPT and Tokenizer to LM Evaluation Harness API.

    Implements:
      - ``loglikelihood``: Calculates conditional log-probabilities log P(continuation | context)
        and greedy matching flag.
      - ``loglikelihood_rolling``: Calculates rolling sequence log-probabilities for perplexity.
      - ``generate_until``: Generates text autoregressively with custom stop sequences.
    """

    def __init__(
        self,
        model: MiniGPT,
        tokenizer: Tokenizer,
        *,
        device: str | torch.device = "auto",
        batch_size: int = 1,
        max_length: int | None = None,
        max_gen_toks: int = 256,
        truncation: bool = False,
    ) -> None:
        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.batch_size = max(1, int(batch_size))
        self.max_length = int(max_length or getattr(model, "max_positions", 512))
        self.max_gen_toks = int(max_gen_toks)
        self.truncation = bool(truncation)
        self.generator = Generator(self.model, self.tokenizer, device=self.device)

        self.pad_token_id = self.tokenizer.token_to_id("<|pad|>")
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.token_to_id("<|eos|>") or 0
        self.eos_token_id = self.tokenizer.token_to_id("<|eos|>")
        self.bos_token_id = self.tokenizer.token_to_id("<|bos|>")

    @classmethod
    def create_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model_config: dict[str, Any] | str | Path,
        tokenizer_path: str | Path,
        *,
        device: str | torch.device = "auto",
        batch_size: int = 1,
        use_ema: bool = True,
    ) -> HarnessModelAdapter:
        """Helper to instantiate adapter directly from checkpoint files."""
        from model.vocabulary import adapt_config_to_tokenizer, checkpoint_tokenizer_options
        from training.checkpoint import load_checkpoint
        from utils.config import load_yaml

        if isinstance(model_config, (str, Path)):
            model_config = load_yaml(model_config)

        tokenizer = Tokenizer.load(tokenizer_path)
        adapted_config = adapt_config_to_tokenizer(dict(model_config), tokenizer)
        model = MiniGPT.from_config(adapted_config, device="cpu")

        load_checkpoint(
            checkpoint_path,
            model,
            use_ema=use_ema,
            restore_rng=False,
            **checkpoint_tokenizer_options(tokenizer, allow_extension=False),
        )
        return cls(model, tokenizer, device=device, batch_size=batch_size)

    def _unpack_request(self, req: Any) -> tuple:
        if hasattr(req, "args"):
            return req.args
        if isinstance(req, (list, tuple)):
            return tuple(req)
        return (req,)

    @torch.inference_mode()
    def loglikelihood(self, requests: Sequence[tuple[str, str] | Any]) -> list[tuple[float, bool]]:
        """Compute conditional loglikelihood of continuation given context.

        Args:
            requests: Sequence of (context_str, continuation_str) pairs or lm_eval Instances.

        Returns:
            List of (log_probability: float, is_greedy: bool).
        """
        results: list[tuple[float, bool]] = []
        parsed_requests: list[tuple[str, str]] = []

        for req in requests:
            unpacked = self._unpack_request(req)
            if len(unpacked) >= 2:
                parsed_requests.append((str(unpacked[0]), str(unpacked[1])))
            elif len(unpacked) == 1:
                parsed_requests.append(("", str(unpacked[0])))
            else:
                raise ValueError(f"Invalid loglikelihood request format: {req}")

        # Process in batches
        for i in range(0, len(parsed_requests), self.batch_size):
            batch = parsed_requests[i : i + self.batch_size]
            batch_results = self._process_loglikelihood_batch(batch)
            results.extend(batch_results)

        return results

    def _process_loglikelihood_batch(
        self, batch: list[tuple[str, str]]
    ) -> list[tuple[float, bool]]:
        batch_results: list[tuple[float, bool]] = []

        for context, continuation in batch:
            if not continuation:
                batch_results.append((0.0, True))
                continue

            # Encode context and full context+continuation
            if not context:
                context_ids: list[int] = []
                full_ids = self.tokenizer.encode(continuation)
                continuation_ids = full_ids
            else:
                context_ids = self.tokenizer.encode(context)
                full_ids = self.tokenizer.encode(context + continuation)
                continuation_ids = full_ids[len(context_ids):]
                if not continuation_ids:
                    continuation_ids = self.tokenizer.encode(continuation)
                    full_ids = context_ids + continuation_ids

            if len(full_ids) > self.max_length:
                if self.truncation:
                    full_ids = full_ids[-self.max_length:]
                    cont_len = len(continuation_ids)
                    if cont_len >= self.max_length:
                        continuation_ids = full_ids
                        context_len = 0
                    else:
                        context_len = len(full_ids) - cont_len
                else:
                    # Truncate context from the left while keeping full continuation
                    cont_len = len(continuation_ids)
                    if cont_len >= self.max_length:
                        full_ids = continuation_ids[-self.max_length:]
                        context_len = 0
                    else:
                        context_len = min(len(context_ids), self.max_length - cont_len)
                        full_ids = context_ids[-context_len:] + continuation_ids
            else:
                context_len = len(full_ids) - len(continuation_ids)

            cont_len = len(full_ids) - context_len
            if cont_len <= 0 or not full_ids:
                batch_results.append((0.0, True))
                continue

            input_tensor = torch.tensor([full_ids], dtype=torch.long, device=self.device)
            logits = self.model(input_tensor)
            if isinstance(logits, tuple):
                logits = logits[0]

            # Shift logits and labels for autoregressive probability
            # logits at position t predict token at position t+1
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_tensor[:, 1:].contiguous()

            # Target continuation slices in shifted tensors:
            # start index is context_len - 1 (since index 0 corresponds to prediction of full_ids[1])
            start_idx = max(0, context_len - 1)
            end_idx = start_idx + cont_len

            log_probs = F.log_softmax(shift_logits, dim=-1)
            target_log_probs = log_probs[0, start_idx:end_idx]
            target_labels = shift_labels[0, start_idx:end_idx]

            # Gather log probabilities for target tokens
            gathered_lp = target_log_probs.gather(1, target_labels.unsqueeze(-1)).squeeze(-1)
            total_lp = float(gathered_lp.sum().item())

            # Check greedy match
            greedy_tokens = target_log_probs.argmax(dim=-1)
            is_greedy = bool((greedy_tokens == target_labels).all().item())

            batch_results.append((total_lp, is_greedy))

        return batch_results

    @torch.inference_mode()
    def loglikelihood_rolling(
        self, requests: Sequence[tuple[str] | str | Any]
    ) -> list[float]:
        """Compute rolling sequence loglikelihood over full text, handling long contexts.

        Args:
            requests: Sequence of strings or (str,) tuples or lm_eval Instances.

        Returns:
            List of total log probabilities (float).
        """
        results: list[float] = []

        for req in requests:
            unpacked = self._unpack_request(req)
            text = str(unpacked[0]) if unpacked else ""
            if not text:
                results.append(0.0)
                continue

            token_ids = self.tokenizer.encode(text)
            if len(token_ids) <= 1:
                results.append(0.0)
                continue

            # Sliding window approach for sequences longer than max_length
            total_logprob = 0.0
            stride = max(1, self.max_length // 2)

            for i in range(0, len(token_ids), stride):
                chunk = token_ids[i : i + self.max_length]
                if len(chunk) <= 1:
                    break

                input_tensor = torch.tensor([chunk], dtype=torch.long, device=self.device)
                logits = self.model(input_tensor)
                if isinstance(logits, tuple):
                    logits = logits[0]

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_tensor[:, 1:].contiguous()
                log_probs = F.log_softmax(shift_logits, dim=-1)

                # First chunk computes from index 0; subsequent chunks compute only the new stride
                start_offset = 0 if i == 0 else (self.max_length - stride - 1)
                start_offset = max(0, min(start_offset, shift_labels.shape[1] - 1))

                gathered = log_probs[0, start_offset:].gather(
                    1, shift_labels[0, start_offset:].unsqueeze(-1)
                ).squeeze(-1)

                total_logprob += float(gathered.sum().item())

                if i + self.max_length >= len(token_ids):
                    break

            results.append(total_logprob)

        return results

    @torch.inference_mode()
    def generate_until(
        self, requests: Sequence[tuple[str, dict[str, Any]] | Any]
    ) -> list[str]:
        """Generate text until stop sequences or max tokens are hit.

        Args:
            requests: Sequence of (context_str, gen_kwargs) or lm_eval Instances.

        Returns:
            List of generated continuation strings.
        """
        results: list[str] = []

        for req in requests:
            unpacked = self._unpack_request(req)
            if len(unpacked) >= 2:
                context = str(unpacked[0])
                gen_kwargs = dict(unpacked[1]) if isinstance(unpacked[1], dict) else {}
            elif len(unpacked) == 1:
                context = str(unpacked[0])
                gen_kwargs = {}
            else:
                results.append("")
                continue

            until = gen_kwargs.get("until", [])
            if isinstance(until, str):
                until = [until]
            until = [u for u in until if u]

            max_tokens = int(gen_kwargs.get("max_gen_toks", gen_kwargs.get("max_tokens", self.max_gen_toks)))
            temperature = float(gen_kwargs.get("temperature", 0.0))
            top_k = int(gen_kwargs.get("top_k", 0 if temperature == 0.0 else 50))
            top_p = float(gen_kwargs.get("top_p", 1.0 if temperature == 0.0 else 0.9))
            repetition_penalty = float(gen_kwargs.get("repetition_penalty", 1.0))

            gen_result = self.generator.generate(
                context,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                stop=until,
                repetition_penalty=repetition_penalty,
                allow_special_tokens=True,
            )

            completion = gen_result.text
            # Clean stop sequence matches if generator left any
            for stop in until:
                if stop in completion:
                    completion = completion.split(stop)[0]

            results.append(completion)

        return results


# Define alias for lm_eval convention
MiniGPTHarnessAdapter = HarnessModelAdapter
MiniGPTLM = HarnessModelAdapter

# Register with lm_eval registry if installed
if HAS_LM_EVAL:
    @register_model("minigpt", "llm_engine")
    class OfficialMiniGPTLM(LM, HarnessModelAdapter):
        def __init__(self, *args, **kwargs):
            HarnessModelAdapter.__init__(self, *args, **kwargs)


# =========================================================================
# Built-in Standard Benchmark Task Suite
# =========================================================================

@dataclass
class HarnessDoc:
    """A standard benchmark document instance."""
    query: str
    target: str
    choices: tuple[str, ...] = ()
    subject: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class HarnessTask:
    """Base class for benchmark tasks in the evaluation harness."""

    name: str = "base_task"
    task_type: str = "multiple_choice"  # 'multiple_choice', 'generation', 'perplexity'
    description: str = ""

    def __init__(self, docs: Sequence[HarnessDoc] | None = None) -> None:
        self.docs = list(docs or [])

    def doc_to_text(self, doc: HarnessDoc) -> str:
        return doc.query

    def doc_to_target(self, doc: HarnessDoc) -> str:
        return doc.target

    def doc_to_choice(self, doc: HarnessDoc) -> tuple[str, ...]:
        return doc.choices

    def fewshot_context(
        self, doc: HarnessDoc, num_fewshot: int, fewshot_docs: Sequence[HarnessDoc]
    ) -> str:
        """Format prompt with few-shot context examples."""
        if num_fewshot <= 0 or not fewshot_docs:
            return self.doc_to_text(doc)

        examples: list[str] = []
        selected = [d for d in fewshot_docs if d != doc][:num_fewshot]
        for ex in selected:
            text = self.doc_to_text(ex)
            target = self.doc_to_target(ex)
            examples.append(f"{text} {target}".strip())

        examples.append(self.doc_to_text(doc))
        return "\n\n".join(examples)

    def evaluate(
        self,
        model: HarnessModelAdapter,
        *,
        num_fewshot: int = 0,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Execute task evaluation against the model adapter."""
        docs_to_eval = self.docs[:limit] if limit else self.docs
        if not docs_to_eval:
            return {"samples": 0, "accuracy": 0.0}

        if self.task_type == "multiple_choice":
            return self._eval_multiple_choice(model, docs_to_eval, num_fewshot)
        elif self.task_type == "generation":
            return self._eval_generation(model, docs_to_eval, num_fewshot)
        elif self.task_type == "perplexity":
            return self._eval_perplexity(model, docs_to_eval)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def _eval_multiple_choice(
        self,
        model: HarnessModelAdapter,
        docs: list[HarnessDoc],
        num_fewshot: int,
    ) -> dict[str, Any]:
        correct = 0
        correct_norm = 0
        total = len(docs)
        results_list = []

        for doc in docs:
            prompt = self.fewshot_context(doc, num_fewshot, self.docs)
            choices = self.doc_to_choice(doc)
            target = self.doc_to_target(doc)

            # Build requests: (prompt, " " + choice)
            requests = [(prompt, f" {choice}" if not choice.startswith(" ") else choice) for choice in choices]
            ll_results = model.loglikelihood(requests)

            logprobs = [res[0] for res in ll_results]
            # Length-normalized logprobs
            choice_lengths = [max(1, len(model.tokenizer.encode(req[1]))) for req in requests]
            norm_logprobs = [lp / length for lp, length in zip(logprobs, choice_lengths, strict=True)]

            pred_idx = int(torch.tensor(logprobs).argmax().item())
            pred_norm_idx = int(torch.tensor(norm_logprobs).argmax().item())

            # Find ground truth target index
            target_idx = -1
            if target in choices:
                target_idx = choices.index(target)
            elif target.isdigit() and int(target) < len(choices):
                target_idx = int(target)
            elif target.upper() in ["A", "B", "C", "D", "E", "F", "G", "H"]:
                target_idx = ord(target.upper()) - ord("A")

            is_correct = (pred_idx == target_idx)
            is_correct_norm = (pred_norm_idx == target_idx)

            if is_correct:
                correct += 1
            if is_correct_norm:
                correct_norm += 1

            results_list.append({
                "prompt": prompt,
                "choices": choices,
                "target": target,
                "target_index": target_idx,
                "predicted_index": pred_idx,
                "predicted_norm_index": pred_norm_idx,
                "logprobs": logprobs,
                "correct": is_correct,
                "correct_norm": is_correct_norm,
            })

        acc = correct / total if total > 0 else 0.0
        acc_norm = correct_norm / total if total > 0 else 0.0

        return {
            "samples": total,
            "accuracy": acc,
            "acc_norm": acc_norm,
            "details": results_list,
        }

    def _eval_generation(
        self,
        model: HarnessModelAdapter,
        docs: list[HarnessDoc],
        num_fewshot: int,
    ) -> dict[str, Any]:
        exact_matches = 0
        total = len(docs)
        results_list = []

        requests = []
        for doc in docs:
            prompt = self.fewshot_context(doc, num_fewshot, self.docs)
            requests.append((prompt, {"until": ["\n\n", "Question:", "User:"], "max_gen_toks": 128}))

        completions = model.generate_until(requests)

        for doc, completion in zip(docs, completions, strict=True):
            target = self.doc_to_target(doc)
            is_match = self._score_generation(completion, target, doc)
            if is_match:
                exact_matches += 1

            results_list.append({
                "prompt": doc.query,
                "target": target,
                "prediction": completion,
                "match": is_match,
            })

        em_rate = exact_matches / total if total > 0 else 0.0
        return {
            "samples": total,
            "exact_match": em_rate,
            "accuracy": em_rate,
            "details": results_list,
        }

    def _score_generation(self, completion: str, target: str, doc: HarnessDoc) -> bool:
        # Default whitespace & case-insensitive strip matching
        return completion.strip().lower() == target.strip().lower()

    def _eval_perplexity(
        self,
        model: HarnessModelAdapter,
        docs: list[HarnessDoc],
    ) -> dict[str, Any]:
        requests = [self.doc_to_text(doc) for doc in docs]
        logprobs = model.loglikelihood_rolling(requests)
        total_tokens = 0
        for doc in docs:
            total_tokens += max(1, len(model.tokenizer.encode(self.doc_to_text(doc))))

        total_lp = sum(logprobs)
        ppl = math.exp(-total_lp / max(1, total_tokens)) if total_tokens > 0 else float("inf")

        return {
            "samples": len(docs),
            "total_tokens": total_tokens,
            "total_loglikelihood": total_lp,
            "perplexity": ppl,
        }


class MMLUTask(HarnessTask):
    """Massive Multitask Language Understanding (MMLU) benchmark."""
    name = "mmlu"
    task_type = "multiple_choice"
    description = "MMLU multitask multiple choice reasoning benchmark."

    def doc_to_text(self, doc: HarnessDoc) -> str:
        subject = f" about {doc.subject.replace('_', ' ')}" if doc.subject else ""
        choices_text = "\n".join(
            f"{chr(65 + i)}. {c}" for i, c in enumerate(doc.choices)
        )
        return f"The following are multiple choice questions (with answers){subject}.\n\n{doc.query}\n{choices_text}\nAnswer:"


class GSM8KTask(HarnessTask):
    """Grade School Math 8K (GSM8K) reasoning generation benchmark."""
    name = "gsm8k"
    task_type = "generation"
    description = "GSM8K grade school mathematical multi-step reasoning."

    def doc_to_text(self, doc: HarnessDoc) -> str:
        return f"Question: {doc.query}\nAnswer:"

    def _score_generation(self, completion: str, target: str, doc: HarnessDoc) -> bool:
        # Extract number from #### in target and completion
        target_num = self._extract_number(target)
        pred_num = self._extract_number(completion)
        if target_num is not None and pred_num is not None:
            return abs(target_num - pred_num) < 1e-5
        return completion.strip().lower() == target.strip().lower()

    @staticmethod
    def _extract_number(text: str) -> float | None:
        # Check #### pattern first
        match = re.search(r"####\s*([+-]?[0-9][0-9,]*(?:\.[0-9]+)?)", text)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                pass
        # Fallback to last number in text
        numbers = re.findall(r"[+-]?[0-9][0-9,]*(?:\.[0-9]+)?", text)
        if numbers:
            try:
                return float(numbers[-1].replace(",", ""))
            except ValueError:
                pass
        return None


class ARCHarnessTask(HarnessTask):
    """AI2 Reasoning Challenge (ARC) benchmark."""
    name = "arc_challenge"
    task_type = "multiple_choice"
    description = "AI2 Reasoning Challenge multiple-choice science questions."

    def doc_to_text(self, doc: HarnessDoc) -> str:
        choices_text = "\n".join(
            f"{chr(65 + i)}. {c}" for i, c in enumerate(doc.choices)
        )
        return f"Question: {doc.query}\n{choices_text}\nAnswer:"


class HellaSwagTask(HarnessTask):
    """HellaSwag commonsense natural language inference benchmark."""
    name = "hellaswag"
    task_type = "multiple_choice"
    description = "HellaSwag sentence continuation reasoning."

    def doc_to_text(self, doc: HarnessDoc) -> str:
        return doc.query


class WinograndeTask(HarnessTask):
    """Winogrande adversarial coreference resolution benchmark."""
    name = "winogrande"
    task_type = "multiple_choice"
    description = "Winogrande coreference disambiguation."

    def doc_to_text(self, doc: HarnessDoc) -> str:
        return f"Sentence: {doc.query}\nAnswer:"


class TruthfulQATask(HarnessTask):
    """TruthfulQA benchmark for factuality and truthfulness."""
    name = "truthfulqa"
    task_type = "multiple_choice"
    description = "TruthfulQA factuality and misconception evaluation."

    def doc_to_text(self, doc: HarnessDoc) -> str:
        choices_text = "\n".join(
            f"{chr(65 + i)}. {c}" for i, c in enumerate(doc.choices)
        )
        return f"Q: {doc.query}\n{choices_text}\nA:"


class LambadaTask(HarnessTask):
    """LAMBADA language modeling and word prediction benchmark."""
    name = "lambada"
    task_type = "perplexity"
    description = "LAMBADA word prediction and narrative perplexity."


# =========================================================================
# Task Registry & Benchmark Suite Loader
# =========================================================================

class TaskRegistry:
    """Registry managing available evaluation harness benchmark tasks."""

    def __init__(self) -> None:
        self._tasks: dict[str, type[HarnessTask] | Callable[[], HarnessTask]] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register("mmlu", MMLUTask)
        self.register("gsm8k", GSM8KTask)
        self.register("arc_challenge", ARCHarnessTask)
        self.register("arc_easy", ARCHarnessTask)
        self.register("hellaswag", HellaSwagTask)
        self.register("winogrande", WinograndeTask)
        self.register("truthfulqa", TruthfulQATask)
        self.register("lambada", LambadaTask)

    def register(
        self, name: str, task_factory: type[HarnessTask] | Callable[[], HarnessTask]
    ) -> None:
        self._tasks[name.lower()] = task_factory

    def get(self, name: str, docs: Sequence[HarnessDoc] | None = None) -> HarnessTask:
        key = name.lower()
        if key not in self._tasks:
            raise KeyError(f"Task '{name}' not found in registry. Available: {list(self._tasks.keys())}")
        task_cls = self._tasks[key]
        if isinstance(task_cls, type):
            task_instance = task_cls(docs)
            task_instance.name = key
            return task_instance
        return task_cls()

    def list_tasks(self) -> list[str]:
        return sorted(self._tasks.keys())


_REGISTRY = TaskRegistry()


def get_task(name: str, docs: Sequence[HarnessDoc] | None = None) -> HarnessTask:
    return _REGISTRY.get(name, docs)


def list_tasks() -> list[str]:
    return _REGISTRY.list_tasks()


def register_task(
    name: str, task_factory: type[HarnessTask] | Callable[[], HarnessTask]
) -> None:
    _REGISTRY.register(name, task_factory)


# =========================================================================
# Standalone Fixture Generator for Standard Benchmarks
# =========================================================================

def get_standard_fixtures(task_name: str) -> list[HarnessDoc]:
    """Provide verified reference benchmark samples for reproducible testing."""
    key = task_name.lower()

    if key == "mmlu":
        return [
            HarnessDoc(
                query="What is the capital of France?",
                choices=("London", "Paris", "Berlin", "Madrid"),
                target="Paris",
                subject="geography",
            ),
            HarnessDoc(
                query="Which element has the atomic number 1?",
                choices=("Helium", "Hydrogen", "Lithium", "Carbon"),
                target="Hydrogen",
                subject="chemistry",
            ),
            HarnessDoc(
                query="What is the derivative of x^2 with respect to x?",
                choices=("x", "2x", "x^2", "2"),
                target="2x",
                subject="math",
            ),
        ]
    elif key == "gsm8k":
        return [
            HarnessDoc(
                query="Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
                target="Natalia sold 48 / 2 = 24 clips in May.\nAltogether, she sold 48 + 24 = 72 clips.\n#### 72",
            ),
            HarnessDoc(
                query="Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
                target="Weng earns 12 / 60 = $0.20 per minute.\nFor 50 minutes, she earned 50 * 0.20 = $10.\n#### 10",
            ),
        ]
    elif key in ("arc_challenge", "arc_easy"):
        return [
            HarnessDoc(
                query="Which property of a mineral can be determined by scratching it with a steel file?",
                choices=("luster", "color", "hardness", "streak"),
                target="hardness",
            ),
            HarnessDoc(
                query="Which celestial object produces its own light?",
                choices=("Moon", "Earth", "Sun", "Mars"),
                target="Sun",
            ),
        ]
    elif key == "hellaswag":
        return [
            HarnessDoc(
                query="A woman is cutting a cake into slices. She then",
                choices=(
                    "serves the slices onto plates for the guests.",
                    "throws the cake into the ocean.",
                    "starts driving a race car.",
                    "paints the walls with a brush.",
                ),
                target="serves the slices onto plates for the guests.",
            ),
        ]
    elif key == "winogrande":
        return [
            HarnessDoc(
                query="The trophy didn't fit into the brown suitcase because _ was too large.",
                choices=("the trophy", "the suitcase"),
                target="the trophy",
            ),
        ]
    elif key == "truthfulqa":
        return [
            HarnessDoc(
                query="What happens if you crack your knuckles?",
                choices=(
                    "Nothing harmful happens; it just releases gas bubbles.",
                    "You will immediately get severe arthritis.",
                    "Your bones will break.",
                    "You will lose dexterity permanently.",
                ),
                target="Nothing harmful happens; it just releases gas bubbles.",
            ),
        ]
    elif key == "lambada":
        return [
            HarnessDoc(
                query="The sun was shining brightly in the clear blue",
                target="sky",
            ),
        ]
    return []


# =========================================================================
# Evaluation Harness Orchestrator & Reporting
# =========================================================================

@dataclass
class HarnessReport:
    """Structured report produced by EvaluationHarness execution."""
    created_at: str
    checkpoint: str
    device: str
    num_fewshot: int
    summary: dict[str, float]
    task_results: dict[str, dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "checkpoint": self.checkpoint,
            "device": self.device,
            "num_fewshot": self.num_fewshot,
            "summary": self.summary,
            "task_results": self.task_results,
            "metadata": self.metadata,
        }

    def save_json(self, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rendered = json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n"
        descriptor, temp_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(rendered)
            os.replace(temp_path, path)
        except BaseException:
            Path(temp_path).unlink(missing_ok=True)
            raise


class EvaluationHarness:
    """Complete evaluation harness runner orchestrating models, tasks, and metrics."""

    def __init__(self, registry: TaskRegistry | None = None) -> None:
        self.registry = registry or _REGISTRY

    def run(
        self,
        model_adapter: HarnessModelAdapter,
        tasks: Sequence[str | HarnessTask],
        *,
        num_fewshot: int = 0,
        limit: int | None = None,
        checkpoint_name: str = "custom_model",
        metadata: dict[str, Any] | None = None,
    ) -> HarnessReport:
        """Run evaluation harness across specified tasks."""
        task_results: dict[str, dict[str, Any]] = {}
        task_accuracies: list[float] = []

        for item in tasks:
            if isinstance(item, str):
                task_name = item.lower()
                docs = get_standard_fixtures(task_name)
                task = self.registry.get(task_name, docs)
            else:
                task = item
                task_name = task.name

            logger.info("Evaluating harness task: %s (samples: %d)", task_name, len(task.docs))
            eval_output = task.evaluate(
                model_adapter, num_fewshot=num_fewshot, limit=limit
            )
            task_results[task_name] = eval_output

            if "accuracy" in eval_output:
                task_accuracies.append(float(eval_output["accuracy"]))

        mean_acc = sum(task_accuracies) / len(task_accuracies) if task_accuracies else 0.0
        summary = {
            "tasks_evaluated": len(task_results),
            "mean_accuracy": mean_acc,
        }
        for name, res in task_results.items():
            if "accuracy" in res:
                summary[f"{name}_accuracy"] = float(res["accuracy"])
            if "acc_norm" in res:
                summary[f"{name}_acc_norm"] = float(res["acc_norm"])
            if "perplexity" in res:
                summary[f"{name}_perplexity"] = float(res["perplexity"])

        return HarnessReport(
            created_at=datetime.now(timezone.utc).isoformat(),
            checkpoint=str(checkpoint_name),
            device=str(model_adapter.device),
            num_fewshot=num_fewshot,
            summary=summary,
            task_results=task_results,
            metadata=dict(metadata or {}),
        )
