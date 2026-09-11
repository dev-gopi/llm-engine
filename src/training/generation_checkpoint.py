"""Preserve inference weights selected by a fixed generation evaluation."""
from pathlib import Path
import math

import torch

from training.checkpoint import save_checkpoint


def save_best_generation(path, model, *, accuracy, evaluation_signature, step, metadata):
    """Save strict improvements; ties retain the earlier model, including on restart.

    Call while the evaluated parameters (for example EMA) are installed. This
    is an inference checkpoint, without optimizer or training resume state.
    """
    accuracy = float(accuracy)
    if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
        raise ValueError('generation accuracy must be finite and between zero and one')
    path = Path(path)
    if path.exists():
        previous = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        saved = previous.get('metadata', {})
        if saved.get('evaluation_signature') != evaluation_signature:
            raise ValueError('generation evaluation changed; choose a new best_output path')
        if accuracy <= float(saved['generation_accuracy']):
            return False
        del previous
    save_checkpoint(path, model, step=step, metadata={
        **metadata, 'generation_accuracy': accuracy,
        'evaluation_signature': evaluation_signature, 'inference_only': True,
    })
    return True
