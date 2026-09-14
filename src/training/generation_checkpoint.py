"""Preserve inference weights selected by a fixed generation evaluation."""
from pathlib import Path
import math

import torch

from training.checkpoint import save_checkpoint


def save_best_generation(path, model, *, accuracy, evaluation_signature, step, metadata,
                         case_scores=None, preserve_passed=False):
    """Save strict improvements; ties retain the earlier model, including on restart.

    Call while the evaluated parameters (for example EMA) are installed. This
    is an inference checkpoint, without optimizer or training resume state.
    """
    accuracy = float(accuracy)
    if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
        raise ValueError('generation accuracy must be finite and between zero and one')
    if preserve_passed and (not case_scores or any(value not in (0, 1) for value in case_scores.values())):
        raise ValueError('preserve_passed requires nonempty binary case_scores')
    path = Path(path)
    if path.exists():
        previous = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        saved = previous.get('metadata', {})
        if saved.get('evaluation_signature') != evaluation_signature:
            raise ValueError('generation evaluation changed; choose a new best_output path')
        if preserve_passed:
            previous_scores = saved.get('case_scores', {})
            if previous_scores.keys() != case_scores.keys():
                raise ValueError('retention case coverage changed; choose a new best_output path')
            if any(case_scores[key] < value for key, value in previous_scores.items()):
                return False
        if accuracy <= float(saved['generation_accuracy']):
            return False
        del previous
    save_checkpoint(path, model, step=step, metadata={
        **metadata, 'generation_accuracy': accuracy,
        'evaluation_signature': evaluation_signature, 'inference_only': True,
        **({'case_scores': dict(case_scores)} if case_scores is not None else {}),
    })
    return True
