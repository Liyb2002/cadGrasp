from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,os,subprocess,time,hashlib,shutil,threading
ROOT=Path('/home/yli581/Desktop/cadGrasp');BASE=ROOT/'slides/Co-optimize/data/results/B';SYSTEM=BASE/'data/system_pso_comparison_all52_20261013_01';FROZEN=SYSTEM/'frozen_sources'
old=json.loads((SYSTEM/'previous_evaluation.json').read_text());cases=sorted(old['cases'],key=lambda r:(-r['object_pose_count'],r['id']))
assert len(cases)==52
policies={'pso':'pso_all52_20261013_01','differential':'differential_control_all52_20261013_01'}
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.writing.json');temp.write_text(json.dumps(d,indent=2)+'\n');temp.replace(p)
start=time.perf_counter();batches={};locks={p:threading.Lock()for p in policies}
for policy,name in policies.items():
 folder=BASE/'data'/name;folder.mkdir(exist_ok=False)
 options=dict(output_name=name,iterations=8,frontier_width=32,search_policy='population_soft',oracle_work_mode='normal',initialization_budget=0,warm_start_experiments=[],warm_checkpoint_experiment=None,initialization_seeds=None,resume=None,allow_oracle_fallback=True,allow_coupled_adjustment=True,allow_group_seating=True,allow_joint_reseating=True,allow_placement_guidance=True,allow_height_repair=True,stop_when_feasible=True,allow_native_sliding=False,allow_geometry_validation=False,require_connected_core=True,population_continuous_policy=policy)
 batches[policy]=dict(complete=False,sets=[r['id']for r in cases],group_count=52,options=options,results=[],passed=0,jobs=6,concurrent_other_policy_jobs=6,source_snapshot=str(FROZEN.relative_to(ROOT)),same_physical_sources=True)
 save(folder/'batch.json',batches[policy])
def one(policy,case):
 name=policies[policy];folder=BASE/'data'/name;label=case['id'].replace('/','__');result=folder/'workers'/(label+'.json');payload=folder/'payloads'/(label+'.json');log=folder/'logs'/(label+'.log')
 save(payload,dict(id=case['id'],options=batches[policy]['options'],frozen=str(FROZEN),result=str(result)));log.parent.mkdir(exist_ok=True)
 began=time.perf_counter()
 with log.open('w')as stream:code=subprocess.call([str(ROOT/'.venv/bin/python'),'/tmp/cadgrasp_pso_step4_worker.py',str(payload)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
 row=json.loads(result.read_text())if result.exists()else dict(id=case['id'],pose_count=case['object_pose_count'],passed=False,experiment=name,error='Worker exit '+str(code))
 row.update(process_wall_seconds=time.perf_counter()-began,worker_exit_code=code,execution_log=str(log.relative_to(ROOT)))
 with locks[policy]:
  b=batches[policy];b['results'].append(row);b['passed']=sum(r['passed']for r in b['results']);b['seconds']=time.perf_counter()-start;b['complete']=len(b['results'])==52
  save(folder/'batch.json',b)
 print(policy,len(batches[policy]['results']),'/52','PASS',batches[policy]['passed'],row['id'],round(row['process_wall_seconds'],2),'s',flush=True)
 return row
# Equal independent six-worker queues, running on the same machine/time window.
pools={policy:ThreadPoolExecutor(max_workers=6)for policy in policies}
futures=[pools[policy].submit(one,policy,case)for case in cases for policy in policies]
for f in as_completed(futures):f.result()
for pool in pools.values():pool.shutdown()
manifest=json.loads((SYSTEM/'frozen_sources.json').read_text())
assert all(hashlib.sha256((ROOT/r['frozen']).read_bytes()).hexdigest()==r['frozen_sha256']for r in manifest['files'])
protected=json.loads((SYSTEM/'protected_originals.json').read_text());changed=[p for p,h in protected.items()if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
save(SYSTEM/'step4_verification.json',dict(passed=not changed,changed=changed,protected_files=len(protected),snapshot_files=len(manifest['files']),seconds=time.perf_counter()-start))
assert not changed
print('BOTH STEP4 CAMPAIGNS COMPLETE',flush=True)
