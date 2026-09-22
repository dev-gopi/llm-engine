"""Make repository and src imports work when a script is executed by path."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"

# Direct execution puts scripts/ at sys.path[0]. Remove it so scripts/tokenize.py
# cannot shadow the Python standard-library tokenize module, then expose both
# the repository root and src package roots for normal imports.
for entry in (SCRIPTS_ROOT, PROJECT_ROOT, SRC_ROOT):
    value = str(entry)
    while value in sys.path:
        sys.path.remove(value)
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
