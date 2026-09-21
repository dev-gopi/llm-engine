from __future__ import annotations
import argparse,json
from pathlib import Path
from evaluation.tokenizer_quality import benchmark_tokenizer
from tokenizer.encoder import Tokenizer

def main():
 p=argparse.ArgumentParser(); p.add_argument('--tokenizer',required=True); p.add_argument('--samples',required=True); p.add_argument('--output',required=True); a=p.parse_args(); t=Tokenizer.load(a.tokenizer); samples=json.loads(Path(a.samples).read_text()); out=benchmark_tokenizer(t,samples); Path(a.output).write_text(json.dumps(out,indent=2,ensure_ascii=False)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
