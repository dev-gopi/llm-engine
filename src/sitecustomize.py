"""Protect third-party imports when this source directory is on ``sys.path``.

This project has a local ``datasets`` package.  When ``PYTHONPATH`` points
directly at ``src/``, it shadows Hugging Face's installed ``datasets``
package, which is imported by optional dependencies such as
``sentence-transformers``.  Python imports ``sitecustomize`` during startup,
so moving this directory after site-packages prevents that collision while
keeping the project's top-level modules importable for legacy launchers.

Launchers should prefer the repository root on ``PYTHONPATH`` (and imports
such as ``src.datasets``) instead of placing ``src/`` directly on the path.
"""

from __future__ import annotations

import os
import sys


_SOURCE_DIRECTORY = os.path.realpath(os.path.dirname(__file__))


def _is_source_directory(entry: str) -> bool:
    return os.path.realpath(entry or os.curdir) == _SOURCE_DIRECTORY


_source_entries = [entry for entry in sys.path if _is_source_directory(entry)]
if _source_entries:
    sys.path[:] = [entry for entry in sys.path if not _is_source_directory(entry)]
    sys.path.extend(_source_entries)
