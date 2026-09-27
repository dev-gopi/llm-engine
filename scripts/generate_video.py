"""Professional video generation CLI: text/image/video conditioning, presets, extension and manifests."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
script_directory=str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve())==script_directory: sys.path.pop(0)
project_src=str(Path(__file__).resolve().parents[1] / 'src')
if project_src not in sys.path: sys.path.insert(0, project_src)
import torch
from media_generation.runtime import VideoGenerator

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config',type=Path,default=Path('configs/video_generation/high_quality.yaml')); p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--prompt',required=True); p.add_argument('--negative-prompt',default=''); p.add_argument('--output',type=Path,default=Path('outputs/video/generated.mp4'))
    p.add_argument('--frames',type=int); p.add_argument('--height',type=int); p.add_argument('--width',type=int); p.add_argument('--fps',type=int); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--preset',choices=['draft','balanced','quality','max_quality']); p.add_argument('--style',choices=['cinematic','photoreal','anime']); p.add_argument('--steps',type=int); p.add_argument('--guidance-scale',type=float)
    p.add_argument('--init-video',type=Path); p.add_argument('--init-image',type=Path); p.add_argument('--strength',type=float,default=1.0); p.add_argument('--segments',type=int,default=1); p.add_argument('--overlap-frames',type=int,default=2)
    p.add_argument('--crf',type=int,default=18); p.add_argument('-n','--num-outputs',type=int,default=1); p.add_argument('--no-metadata',action='store_true'); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args();
    if a.num_outputs<1: raise ValueError('num-outputs must be positive')
    g=VideoGenerator(a.config,a.checkpoint,device=a.device); results=[]
    for i in range(a.num_outputs):
        out=a.output if a.num_outputs==1 else a.output.with_name(f'{a.output.stem}-{i+1}{a.output.suffix}')
        _,meta=g.generate(prompt=a.prompt,negative_prompt=a.negative_prompt,output=out,frames=a.frames,height=a.height,width=a.width,fps=a.fps,seed=None if a.seed is None else a.seed+i,preset=a.preset,style=a.style,steps=a.steps,guidance_scale=a.guidance_scale,init_video=a.init_video,init_image=a.init_image,strength=a.strength,crf=a.crf,metadata=not a.no_metadata,segments=a.segments,overlap_frames=a.overlap_frames)
        results.append(meta)
    print(json.dumps(results,indent=2))
if __name__=='__main__': main()
