from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.embeddings import HashEmbeddingModel, evaluate_retrieval


def main():
    p=argparse.ArgumentParser(); p.add_argument('--queries',required=True); p.add_argument('--documents',required=True); p.add_argument('--relevance',required=True); p.add_argument('--output',required=True); p.add_argument('--dimension',type=int,default=256); a=p.parse_args()
    queries=json.loads(Path(a.queries).read_text()); docs=json.loads(Path(a.documents).read_text()); relevance=[set(x) for x in json.loads(Path(a.relevance).read_text())]
    result=evaluate_retrieval(HashEmbeddingModel(a.dimension),queries,docs,relevance)
    Path(a.output).write_text(json.dumps(result.to_dict(),indent=2,sort_keys=True)+'\n')
if __name__=='__main__': main()
