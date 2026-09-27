#!/usr/bin/env python3
"""Run durable platform workers. Safe to run as one-shot CronJob or long-lived worker."""
from __future__ import annotations
import argparse, asyncio, os
from serving.openai_platform import PlatformStore
from serving.redis_platform_store import RedisPlatformStore
from serving.platform_workers import PlatformWorkers

async def main():
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');p.add_argument('--interval',type=float,default=2.0);p.add_argument('--types',default='vector,batch,finetune');a=p.parse_args()
    redis=os.getenv('GOPI_DISTRIBUTED_STATE_REDIS_URL'); files=os.getenv('GOPI_PLATFORM_FILES_DIR','data/cache/openai-files')
    store=RedisPlatformStore(redis,files,key_prefix=f"{os.getenv('GOPI_DISTRIBUTED_STATE_PREFIX','llm-engine:state')}:platform") if redis else PlatformStore(os.getenv('GOPI_PLATFORM_DB','data/cache/openai-platform.sqlite3'),files)
    workers=PlatformWorkers(store,vector_index_path=os.getenv('GOPI_VECTOR_INDEX_DB','data/cache/vector-index.sqlite3'))
    selected={x.strip() for x in a.types.split(',') if x.strip()}
    while True:
        if 'vector' in selected: await workers.run_vector_once()
        if 'batch' in selected: await workers.run_batch_once()
        if 'finetune' in selected: await workers.run_finetune_once()
        if a.once:return
        await asyncio.sleep(max(.1,a.interval))
if __name__=='__main__': asyncio.run(main())
