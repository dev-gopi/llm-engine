"""Run reproducible JSONL audio/video generation jobs."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
script_directory=str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve())==script_directory: sys.path.pop(0)
project_src=str(Path(__file__).resolve().parents[1] / 'src')
if project_src not in sys.path: sys.path.insert(0, project_src)
from media_generation.runtime import AudioGenerator, VideoGenerator

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--jobs',type=Path,required=True); p.add_argument('--audio-config',type=Path); p.add_argument('--audio-checkpoint',type=Path); p.add_argument('--video-config',type=Path); p.add_argument('--video-checkpoint',type=Path); p.add_argument('--device'); p.add_argument('--continue-on-error',action='store_true'); a=p.parse_args()
    audio=video=None; results=[]
    for line_no,line in enumerate(a.jobs.read_text().splitlines(),1):
        if not line.strip(): continue
        try:
            job=json.loads(line); kind=job.pop('kind'); output=job.pop('output')
            if kind=='audio':
                if audio is None: audio=AudioGenerator(a.audio_config,a.audio_checkpoint,device=a.device)
                _,meta=audio.generate(output=output,**job)
            elif kind=='video':
                if video is None: video=VideoGenerator(a.video_config,a.video_checkpoint,device=a.device)
                _,meta=video.generate(output=output,**job)
            else: raise ValueError('kind must be audio or video')
            results.append({'line':line_no,'ok':True,**meta})
        except Exception as exc:
            results.append({'line':line_no,'ok':False,'error':str(exc)})
            if not a.continue_on_error: raise
    print(json.dumps(results,indent=2))
if __name__=='__main__': main()
