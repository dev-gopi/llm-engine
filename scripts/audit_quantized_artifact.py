from __future__ import annotations
import argparse,json
from pathlib import Path
from inference.quantization import validate_quantized_manifest
from utils.config import load_yaml
p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--model-config',required=True); a=p.parse_args()
manifest=json.loads(Path(a.manifest).read_text()); config=load_yaml(a.model_config); validate_quantized_manifest(manifest,config=config); print(json.dumps({'status':'verified','format':manifest['format'],'architecture':manifest['architecture']},indent=2))
