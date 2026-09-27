"""Professional audio generation CLI: text/audio conditioning, presets, batches, long-form and manifests."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
script_directory=str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve())==script_directory: sys.path.pop(0)
project_src=str(Path(__file__).resolve().parents[1] / 'src')
if project_src not in sys.path: sys.path.insert(0, project_src)
import torch
from media_generation.runtime import AudioGenerator


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/audio_generation/high_quality.yaml'))
    p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--prompt',required=True); p.add_argument('--negative-prompt',default='')
    p.add_argument('--output',type=Path,default=Path('outputs/audio/generated.wav')); p.add_argument('--seconds',type=float); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--preset',choices=['draft','balanced','quality','max_quality']); p.add_argument('--style',choices=['ambient','music','speech','cinematic'])
    p.add_argument('--steps',type=int); p.add_argument('--guidance-scale',type=float); p.add_argument('--init-audio',type=Path); p.add_argument('--strength',type=float,default=1.0)
    p.add_argument('--preserve-start-seconds',type=float); p.add_argument('--preserve-end-seconds',type=float); p.add_argument('--long-form',action='store_true'); p.add_argument('--overlap-seconds',type=float,default=.5)
    p.add_argument('-n','--num-outputs',type=int,default=1); p.add_argument('--no-metadata',action='store_true'); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args();
    if a.num_outputs<1: raise ValueError('num-outputs must be positive')
    g=AudioGenerator(a.config,a.checkpoint,device=a.device); results=[]
    for i in range(a.num_outputs):
        out=a.output if a.num_outputs==1 else a.output.with_name(f'{a.output.stem}-{i+1}{a.output.suffix}')
        path,meta=g.generate(prompt=a.prompt,negative_prompt=a.negative_prompt,output=out,seconds=a.seconds,seed=None if a.seed is None else a.seed+i,preset=a.preset,style=a.style,steps=a.steps,guidance_scale=a.guidance_scale,init_audio=a.init_audio,strength=a.strength,preserve_start_seconds=a.preserve_start_seconds,preserve_end_seconds=a.preserve_end_seconds,metadata=not a.no_metadata,long_form=a.long_form,overlap_seconds=a.overlap_seconds)
        results.append(meta)
    print(json.dumps(results,indent=2))
if __name__=='__main__': main()
