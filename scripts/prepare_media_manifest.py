"""Create deterministic train/validation JSONL manifests from caption sidecars."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--kind',choices=['audio','video'],required=True); p.add_argument('--root',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--validation-percent',type=float,default=2.0); p.add_argument('--caption-suffix',default='.txt'); a=p.parse_args()
    exts={'.wav'} if a.kind=='audio' else {'.mp4','.mov','.mkv','.webm'}
    records=[]
    for media in sorted(p for p in a.root.rglob('*') if p.suffix.lower() in exts):
        caption=media.with_suffix(a.caption_suffix)
        if not caption.is_file(): continue
        text=caption.read_text(encoding='utf-8').strip()
        if not text: continue
        records.append({a.kind:str(media.resolve()),'text':text})
    if not records: raise SystemExit('no captioned media found')
    a.output_dir.mkdir(parents=True,exist_ok=True); train=[]; val=[]
    threshold=max(0,min(100,float(a.validation_percent)))
    for r in records:
        h=int(hashlib.sha256(r[a.kind].encode()).hexdigest()[:8],16)%10000/100
        (val if h<threshold else train).append(r)
    for name,items in [('train',train),('validation',val)]:
        with (a.output_dir/f'{name}.jsonl').open('w',encoding='utf-8') as f:
            for item in items: f.write(json.dumps(item,ensure_ascii=False)+'\n')
    print(json.dumps({'total':len(records),'train':len(train),'validation':len(val)},indent=2))
if __name__=='__main__': main()
