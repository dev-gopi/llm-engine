#!/usr/bin/env python3
import argparse,yaml
from enterprise import DeploymentManager
p=argparse.ArgumentParser();p.add_argument('--name',default='llm-engine');p.add_argument('--image',required=True);p.add_argument('--output',default='deploy/kubernetes.generated.yaml');a=p.parse_args()
docs=DeploymentManager.kubernetes(a.name,a.image)
with open(a.output,'w') as f:
    yaml.safe_dump_all([docs['deployment'],docs['hpa']],f,sort_keys=False)
print(a.output)
