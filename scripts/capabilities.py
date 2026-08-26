"""Print the training features supported by the current PyTorch and hardware."""

from __future__ import annotations

import json
import sys
from pathlib import Path

script_directory = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == script_directory:
    sys.path.pop(0)

from training.capabilities import training_capabilities

print(json.dumps(training_capabilities(), indent=2))
