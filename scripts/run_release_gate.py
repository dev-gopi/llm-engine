from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.release_gate import GateRule, ReleaseGate


def main():
 p=argparse.ArgumentParser(); p.add_argument('--baseline',required=True); p.add_argument('--candidate',required=True); p.add_argument('--output',required=True); a=p.parse_args(); b=json.loads(Path(a.baseline).read_text()); c=json.loads(Path(a.candidate).read_text()); rules=[GateRule(k,v.get('direction','max'),float(v.get('max_regression',0)),True) for k,v in c.get('_gate_rules',{}).items()]; result=ReleaseGate(rules).evaluate(b,c); Path(a.output).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2)); raise SystemExit(0 if result['passed'] else 2)
if __name__=='__main__': main()
