import asyncio, json
from pathlib import Path
from serving.openai_platform import PlatformStore, VectorStoreCreate, VectorStoreFileCreate, BatchCreate, FineTuningJobCreate
from serving.platform_workers import PlatformWorkers
from enterprise import (EnterpriseDB,EntitlementService,ModerationService,ComplianceService,SCIMDirectory,PolicyEngine,TamperEvidentAuditLog,ArtifactAdmission,DeploymentManager,BackupManager)

def test_vector_ingestion_and_search(tmp_path):
    store=PlatformStore(tmp_path/'p.db',tmp_path/'files')
    f=store.put_file('t','doc.txt','assistants',b'alpha beta gamma. distributed vector retrieval works.')
    vs=store.create_vector_store('t',VectorStoreCreate(name='x'))
    attached=store.attach_vector_file('t',vs['id'],f['id']); assert attached['status']=='pending'
    w=PlatformWorkers(store,vector_index_path=tmp_path/'v.db')
    r=asyncio.run(w.run_vector_once()); assert r.succeeded==1
    hits=w.vector_index.search('t',vs['id'],'vector retrieval'); assert hits and hits[0]['file_id']==f['id']

def test_batch_worker_outputs_and_counts(tmp_path):
    store=PlatformStore(tmp_path/'p.db',tmp_path/'files')
    body='\n'.join([json.dumps({'custom_id':'a','url':'/v1/responses','body':{'input':'hi'}}),'{bad'])+'\n'
    f=store.put_file('t','batch.jsonl','batch',body.encode())
    b=store.create_batch('t',BatchCreate(input_file_id=f['id'],endpoint='/v1/responses'))
    w=PlatformWorkers(store,vector_index_path=tmp_path/'v.db')
    out=asyncio.run(w.execute_batch('t',b['id']))
    assert out['status']=='completed'; assert out['request_counts']=={'total':2,'completed':1,'failed':1}; assert out['output_file_id']; assert out['error_file_id']

def test_finetune_worker_with_injected_command(tmp_path):
    store=PlatformStore(tmp_path/'p.db',tmp_path/'files'); f=store.put_file('t','train.jsonl','fine-tune',b'{}\n')
    j=store.create_finetune('t',FineTuningJobCreate(training_file=f['id'],model='base'))
    w=PlatformWorkers(store,vector_index_path=tmp_path/'v.db',finetune_command_builder=lambda *args:['python','-c','print("ok")'])
    result=w.execute_finetune('t',j['id']); assert result['status']=='succeeded'; assert result['fine_tuned_model'].startswith('ft:base:')

def test_enterprise_controls(tmp_path):
    db=EnterpriseDB(tmp_path/'e.db'); ent=EntitlementService(db); ent.set('t','fine_tuning',True); ent.require('t','fine_tuning'); ent.meter('t','tokens',10); assert ent.usage('t')['tokens']==10
    mod=ModerationService().moderate('I will kill'); assert mod.flagged and mod.categories['violence']
    comp=ComplianceService(db,{'t':{'in-east-1'}}); assert comp.check_region('t','in-east-1') and not comp.check_region('t','us-east-1'); hold=comp.place_hold('t','case'); assert comp.under_hold('t'); comp.release_hold('t',hold)
    scim=SCIMDirectory(db); u=scim.upsert_user('t','a@example.com',displayName='A'); assert scim.users('t')[0]['id']==u['id']
    assert PolicyEngine({'admin':{'*'}}).allow({'admin'},'models.deploy')

def test_audit_supply_chain_deploy_backup(tmp_path):
    audit=TamperEvidentAuditLog(tmp_path/'audit.jsonl','secret'); audit.append({'action':'x'}); audit.append({'action':'y'}); assert audit.verify()
    art=tmp_path/'m.bin'; art.write_bytes(b'model'); adm=ArtifactAdmission(b'k'); m=adm.manifest(art); sig=adm.sign(m); assert adm.verify(art,m,sig)
    k=DeploymentManager.kubernetes('gopi','example/gopi:1'); assert k['deployment']['kind']=='Deployment' and k['hpa']['kind']=='HorizontalPodAutoscaler'
    src=tmp_path/'state.db'; src.write_text('x'); bm=BackupManager(tmp_path/'backups'); b=bm.backup([src]); out=tmp_path/'restore'; bm.restore(b,out); assert (out/'state.db').read_text()=='x'

def test_enterprise_registry_kms_and_failover(tmp_path):
    from enterprise import KMSProvider, WebhookRegistry, ReplicaRegistry, MultiRegionFailover
    assert KMSProvider(local_secret=b'k').sign(b'x')
    wr=WebhookRegistry(tmp_path/'wh.db'); ident=wr.register('t','https://example.test/h','s',['batch.completed']); assert wr.subscriptions('t','batch.completed')[0]['id']==ident; wr.disable('t',ident); assert not wr.subscriptions('t','batch.completed')
    rr=ReplicaRegistry(ttl_seconds=60); rr.heartbeat('a','http://a',region='in'); rr.heartbeat('b','http://b',region='in'); assert rr.route('in')['id'] in {'a','b'}
    assert MultiRegionFailover('in',['sg']).choose({'in':False,'sg':True})=='sg'
