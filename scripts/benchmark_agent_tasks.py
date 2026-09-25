from __future__ import annotations

try:
    from scripts._bootstrap import PROJECT_ROOT  # noqa: F401
except ModuleNotFoundError:
    from _bootstrap import PROJECT_ROOT  # noqa: F401

import argparse
import json
from pathlib import Path

from evaluation.benchmarks import AgentTaskResult, evaluate_agent_tasks

p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--output',required=True); a=p.parse_args()
rows=json.loads(Path(a.input).read_text()); result=evaluate_agent_tasks([AgentTaskResult(**row) for row in rows]); Path(a.output).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
