from __future__ import annotations
import argparse,json
from pathlib import Path
from training.promotion import CheckpointPromoter,CheckpointCandidate,MetricRule

def main():
 p=argparse.ArgumentParser(); p.add_argument('manifest'); p.add_argument('--output',default='reports/checkpoint_promotion.json'); a=p.parse_args()
 data=json.loads(Path(a.manifest).read_text()); rules=[MetricRule(r['name'],r.get('direction','max'),float(r.get('weight',1)),float(r.get('max_regression',0)),bool(r.get('protected',False))) for r in data['rules']]
 candidates=[CheckpointCandidate(x['checkpoint'],x['metrics'],x.get('evidence',{})) for x in data['candidates']]
 d=CheckpointPromoter(rules,baseline=data.get('baseline')).decide(candidates); out={"selected":d.selected,"promoted":d.promoted,"scores":d.scores,"failures":d.failures}
 Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
