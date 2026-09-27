"""Redis-backed OpenAI platform metadata for multi-replica deployments.

File payloads still live in a configured shared filesystem/object-mount path;
all mutable lifecycle metadata is stored in Redis and tenant-indexed.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from .openai_platform import BatchCreate, FineTuningJobCreate, VectorStoreCreate

try:
    import redis
except ImportError:  # pragma: no cover
    redis = None


class RedisPlatformStore:
    def __init__(self, url: str, files_dir: str | Path, *, key_prefix: str = "llm-engine:platform") -> None:
        if redis is None:
            raise RuntimeError("redis package is required for RedisPlatformStore")
        self.client = redis.from_url(url, decode_responses=True)
        self.files_dir = Path(files_dir); self.files_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = key_prefix.rstrip(":")

    @staticmethod
    def _now() -> int: return int(time.time())
    def _obj(self, kind:str, ident:str)->str: return f"{self.prefix}:{kind}:{ident}"
    def _idx(self, kind:str, tenant:str)->str: return f"{self.prefix}:idx:{kind}:{tenant}"
    def _vs_files(self, ident:str)->str: return f"{self.prefix}:vsfiles:{ident}"

    def _write(self, kind:str, ident:str, tenant:str, payload:dict[str,Any], *, score:int|None=None)->None:
        data=json.dumps(payload,separators=(",",":"),ensure_ascii=False)
        pipe=self.client.pipeline(transaction=True); pipe.set(self._obj(kind,ident),data); pipe.zadd(self._idx(kind,tenant),{ident:float(score or self._now())}); pipe.execute()
    def _read(self,kind:str,ident:str)->dict[str,Any]|None:
        raw=self.client.get(self._obj(kind,ident)); return json.loads(raw) if raw else None
    def _list(self,kind:str,tenant:str)->list[str]: return list(self.client.zrevrange(self._idx(kind,tenant),0,-1))

    def put_file(self, tenant:str, filename:str, purpose:str, content:bytes)->dict[str,Any]:
        if len(content)>int(os.getenv("GOPI_FILES_MAX_BYTES",str(100*1024*1024))): raise ValueError("file exceeds configured size limit")
        ident=f"file-{uuid.uuid4().hex}"; safe=Path(filename).name or "upload.bin"; path=self.files_dir/f"{ident}-{safe}"; path.write_bytes(content)
        now=self._now(); digest=hashlib.sha256(content).hexdigest()
        meta={"id":ident,"tenant":tenant,"filename":safe,"purpose":purpose,"bytes":len(content),"sha256":digest,"created_at":now,"path":str(path)}
        self._write("file",ident,tenant,meta,score=now)
        return {"id":ident,"object":"file","bytes":len(content),"created_at":now,"filename":safe,"purpose":purpose,"status":"processed","sha256":digest}

    def get_file(self,tenant:str,file_id:str)->dict[str,Any]|None:
        value=self._read("file",file_id); return value if value and value.get("tenant")==tenant else None
    def list_files(self,tenant:str,purpose:str|None=None)->list[dict[str,Any]]:
        out=[]
        for ident in self._list("file",tenant):
            r=self.get_file(tenant,ident)
            if not r or (purpose and r.get("purpose")!=purpose): continue
            out.append({"id":r["id"],"object":"file","bytes":r["bytes"],"created_at":r["created_at"],"filename":r["filename"],"purpose":r["purpose"],"status":"processed","sha256":r["sha256"]})
        return out
    def delete_file(self,tenant:str,file_id:str)->bool:
        r=self.get_file(tenant,file_id)
        if not r:return False
        Path(r["path"]).unlink(missing_ok=True); pipe=self.client.pipeline(transaction=True); pipe.delete(self._obj("file",file_id)); pipe.zrem(self._idx("file",tenant),file_id)
        for vs in self._list("vs",tenant): pipe.srem(self._vs_files(vs),file_id)
        pipe.execute(); return True

    def create_vector_store(self,tenant:str,req:VectorStoreCreate)->dict[str,Any]:
        ident=f"vs_{uuid.uuid4().hex}"; now=self._now(); payload={"id":ident,"tenant":tenant,"name":req.name,"created_at":now,"expires_at":now+req.expires_after_seconds if req.expires_after_seconds else None,"status":"completed","metadata":req.metadata}
        self._write("vs",ident,tenant,payload,score=now); return self.vector_store(tenant,ident)
    def vector_store(self,tenant:str,ident:str)->dict[str,Any]|None:
        r=self._read("vs",ident)
        if not r or r.get("tenant")!=tenant:return None
        count=int(self.client.scard(self._vs_files(ident))); return {"id":r["id"],"object":"vector_store","name":r["name"],"created_at":r["created_at"],"status":r["status"],"expires_at":r.get("expires_at"),"metadata":r.get("metadata",{}),"file_counts":{"in_progress":0,"completed":count,"failed":0,"cancelled":0,"total":count}}
    def list_vector_stores(self,tenant:str)->list[dict[str,Any]]: return [x for i in self._list("vs",tenant) if (x:=self.vector_store(tenant,i))]
    def delete_vector_store(self,tenant:str,ident:str)->bool:
        if self.vector_store(tenant,ident) is None:return False
        pipe=self.client.pipeline(transaction=True); pipe.delete(self._obj("vs",ident),self._vs_files(ident)); pipe.zrem(self._idx("vs",tenant),ident); pipe.execute(); return True
    def attach_vector_file(self,tenant:str,vector_store_id:str,file_id:str)->dict[str,Any]:
        if self.vector_store(tenant,vector_store_id) is None or self.get_file(tenant,file_id) is None: raise KeyError("vector store or file not found")
        now=self._now(); self.client.sadd(self._vs_files(vector_store_id),file_id); self.client.hset(f"{self.prefix}:vsfile_status:{vector_store_id}",file_id,"pending"); return {"id":file_id,"object":"vector_store.file","vector_store_id":vector_store_id,"status":"pending","created_at":now}

    def create_batch(self,tenant:str,req:BatchCreate)->dict[str,Any]:
        if self.get_file(tenant,req.input_file_id) is None:raise KeyError("input file not found")
        ident=f"batch_{uuid.uuid4().hex}"; now=self._now(); r={"id":ident,"tenant":tenant,"input_file_id":req.input_file_id,"endpoint":req.endpoint,"completion_window":req.completion_window,"status":"validating","created_at":now,"cancelled_at":None,"output_file_id":None,"error_file_id":None,"metadata":req.metadata}
        self._write("batch",ident,tenant,r,score=now); return self.batch(tenant,ident)
    def batch(self,tenant:str,ident:str)->dict[str,Any]|None:
        r=self._read("batch",ident)
        if not r or r.get("tenant")!=tenant:return None
        return {"id":r["id"],"object":"batch","endpoint":r["endpoint"],"input_file_id":r["input_file_id"],"completion_window":r["completion_window"],"status":r["status"],"created_at":r["created_at"],"cancelled_at":r.get("cancelled_at"),"output_file_id":r.get("output_file_id"),"error_file_id":r.get("error_file_id"),"metadata":r.get("metadata",{}),"request_counts":r.get("request_counts",{"total":0,"completed":0,"failed":0})}
    def list_batches(self,tenant:str)->list[dict[str,Any]]: return [x for i in self._list("batch",tenant) if (x:=self.batch(tenant,i))]
    def cancel_batch(self,tenant:str,ident:str)->dict[str,Any]|None:
        r=self._read("batch",ident)
        if not r or r.get("tenant")!=tenant:return None
        r["status"]="cancelled"; r["cancelled_at"]=self._now(); self._write("batch",ident,tenant,r,score=r["created_at"]); return self.batch(tenant,ident)

    def create_finetune(self,tenant:str,req:FineTuningJobCreate)->dict[str,Any]:
        if self.get_file(tenant,req.training_file) is None:raise KeyError("training file not found")
        if req.validation_file and self.get_file(tenant,req.validation_file) is None:raise KeyError("validation file not found")
        ident=f"ftjob-{uuid.uuid4().hex}"; now=self._now(); r={"id":ident,"tenant":tenant,"training_file":req.training_file,"validation_file":req.validation_file,"model":req.model,"status":"queued","created_at":now,"finished_at":None,"suffix":req.suffix,"method":req.method,"hyperparameters":req.hyperparameters,"fine_tuned_model":None,"error":None}
        self._write("ft",ident,tenant,r,score=now); return self.finetune(tenant,ident)
    def finetune(self,tenant:str,ident:str)->dict[str,Any]|None:
        r=self._read("ft",ident)
        if not r or r.get("tenant")!=tenant:return None
        return {"id":r["id"],"object":"fine_tuning.job","model":r["model"],"created_at":r["created_at"],"finished_at":r.get("finished_at"),"fine_tuned_model":r.get("fine_tuned_model"),"status":r["status"],"training_file":r["training_file"],"validation_file":r.get("validation_file"),"suffix":r.get("suffix"),"hyperparameters":r.get("hyperparameters",{}),"error":r.get("error"),"method":r.get("method"),"trained_tokens":r.get("trained_tokens")}
    def list_finetunes(self,tenant:str)->list[dict[str,Any]]: return [x for i in self._list("ft",tenant) if (x:=self.finetune(tenant,i))]
    def cancel_finetune(self,tenant:str,ident:str)->dict[str,Any]|None:
        r=self._read("ft",ident)
        if not r or r.get("tenant")!=tenant:return None
        if r.get("status") in {"queued","running","validating_files"}: r["status"]="cancelled"; r["finished_at"]=self._now(); self._write("ft",ident,tenant,r,score=r["created_at"])
        return self.finetune(tenant,ident)


    def worker_vector_jobs(self,limit:int=16)->list[dict[str,Any]]:
        out=[]
        for key in self.client.scan_iter(match=f"{self.prefix}:vsfile_status:*"):
            vs=str(key).rsplit(':',1)[-1]
            for fid,status in self.client.hgetall(key).items():
                if status in {'pending','failed'}:
                    meta=self._read('vs',vs)
                    if meta: out.append({'tenant':meta['tenant'],'vector_store_id':vs,'file_id':fid,'status':status})
                if len(out)>=limit:return out
        return out
    def worker_update_vector_file(self,tenant,vector_store_id,file_id,status): self.client.hset(f"{self.prefix}:vsfile_status:{vector_store_id}",file_id,status)
    def worker_batch_jobs(self,limit:int=4)->list[dict[str,Any]]:
        out=[]
        for key in self.client.scan_iter(match=f"{self.prefix}:batch:*"):
            r=json.loads(self.client.get(key));
            if r.get('status') in {'validating','queued'}: out.append({'id':r['id'],'tenant':r['tenant']})
            if len(out)>=limit:break
        return out
    def worker_update_batch(self,tenant,ident,*,status,output_file_id=None,error_file_id=None,request_counts=None,error=None):
        r=self._read('batch',ident)
        if not r or r.get('tenant')!=tenant:return
        r['status']=status
        if output_file_id is not None:r['output_file_id']=output_file_id
        if error_file_id is not None:r['error_file_id']=error_file_id
        if request_counts is not None:r['request_counts']=request_counts
        if error is not None:r['worker_error']=error
        self._write('batch',ident,tenant,r,score=r['created_at'])
    def worker_finetune_jobs(self,limit:int=1)->list[dict[str,Any]]:
        out=[]
        for key in self.client.scan_iter(match=f"{self.prefix}:ft:*"):
            r=json.loads(self.client.get(key));
            if r.get('status')=='queued':out.append({'id':r['id'],'tenant':r['tenant']})
            if len(out)>=limit:break
        return out
    def worker_update_finetune(self,tenant,ident,*,status,fine_tuned_model=None,error=None,finished_at=None):
        r=self._read('ft',ident)
        if not r or r.get('tenant')!=tenant:return
        r['status']=status
        if fine_tuned_model is not None:r['fine_tuned_model']=fine_tuned_model
        r['error']=error
        if finished_at is not None:r['finished_at']=finished_at
        self._write('ft',ident,tenant,r,score=r['created_at'])

    def get_idempotent(self,tenant:str,route:str,key:str,request_hash:str):
        ident=hashlib.sha256(f"{tenant}\0{route}\0{key}".encode()).hexdigest(); r=self._read("idem",ident)
        if not r:return None
        if r["request_hash"]!=request_hash:raise ValueError("idempotency key was already used with a different request")
        return int(r["status_code"]),r["payload"]
    def put_idempotent(self,tenant:str,route:str,key:str,request_hash:str,status_code:int,payload:dict[str,Any])->None:
        ident=hashlib.sha256(f"{tenant}\0{route}\0{key}".encode()).hexdigest()
        redis_key=self._obj("idem",ident)
        body=json.dumps({"request_hash":request_hash,"status_code":int(status_code),"payload":payload},separators=(",",":"),ensure_ascii=False)
        script="""
        local old=redis.call('GET', KEYS[1])
        if old then
          local obj=cjson.decode(old)
          if obj.request_hash ~= ARGV[1] then return -1 end
          return 0
        end
        redis.call('SET', KEYS[1], ARGV[2], 'EX', ARGV[3]); return 1
        """
        result=int(self.client.eval(script,1,redis_key,request_hash,body,86400))
        if result < 0: raise ValueError("idempotency key was already used with a different request")


__all__=["RedisPlatformStore"]
