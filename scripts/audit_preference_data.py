from __future__ import annotations

import argparse
import json
from pathlib import Path

from alignment.pipeline import audit_preference_set


def main():
 p=argparse.ArgumentParser(); p.add_argument('input'); p.add_argument('--output',required=True); a=p.parse_args(); records=[json.loads(x) for x in Path(a.input).read_text().splitlines() if x.strip()]; report=audit_preference_set(records); Path(a.output).write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2)); raise SystemExit(0 if report['valid']==report['records'] and not report['duplicates'] else 2)
if __name__=='__main__': main()
