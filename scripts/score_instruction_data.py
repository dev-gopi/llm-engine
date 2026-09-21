from __future__ import annotations
import argparse,json
from pathlib import Path
from datasets.instruction_quality import filter_and_balance

def main():
 p=argparse.ArgumentParser(); p.add_argument('input'); p.add_argument('--output',required=True); p.add_argument('--min-score',type=float,default=.65); a=p.parse_args()
 records=[json.loads(x) for x in Path(a.input).read_text().splitlines() if x.strip()]
 selected,meta=filter_and_balance(records,min_score=a.min_score); Path(a.output).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in selected)); print(json.dumps(meta,indent=2))
if __name__=='__main__': main()
