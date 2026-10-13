from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,os,time,subprocess,shutil,hashlib
ROOT=Path('/home/yli581/Desktop/cadGrasp');BASE=ROOT/'slides/Co-optimize/data/results/B';SYSTEM=BASE/'data/system_pso_comparison_all52_20261013_01';FROZEN=SYSTEM/'frozen_sources'
policies={'differential':'differential_control_all52_20261013_01','pso':'pso_all52_20261013_01'}
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.writing.json');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
while True:
 ready=True
 for exp in policies.values():
  d=json.loads((BASE/'data'/exp/'batch.json').read_text());ready &= d['complete']
 if ready:break
 time.sleep(5)
for policy,exp in policies.items():
 post=exp+'_sculpted';folder=BASE/'data'/post;folder.mkdir(exist_ok=False);source=json.loads((BASE/'data'/exp/'batch.json').read_text());cases=sorted(source['results'],key=lambda r:(-r['pose_count'],r['id']))
 selection=folder/'selection.json';save(selection,dict(complete=True,group_count=52,results=cases,passed=source['passed']))
 rows=[dict(id=r['id'],pose_count=r['pose_count'],experiment=exp,passed=False,status='STEP4_FAIL',skipped_reason=r.get('error','Original force/Oracle/core check failed'))for r in cases if not r['passed']]
 started=time.perf_counter()
 def record():save(folder/'batch.json',dict(complete=len(rows)==52,group_count=52,passed=sum(r['passed']for r in rows),eligible_groups=source['passed'],results=rows,seconds=time.perf_counter()-started,source_summary=str((BASE/'data'/exp/'batch.json').relative_to(ROOT)),policy=policy,postprocess_experiment=post,jobs=6))
 record()
 def one(case):
  label=case['id'].replace('/','__');result=folder/'workers'/(label+'.json');payload=folder/'payloads'/(label+'.json');log=folder/'logs'/(label+'.log');log.parent.mkdir(exist_ok=True)
  config=dict(frozen=str(FROZEN),case=case,selection=str(selection),initial=exp+'_stock',aesthetic=exp+'_rounded',final=post,floor=exp+'_floor',result=str(result))
  save(payload,config)
  for stage in ['step5.1','step5.2','step5.3','evaluation']:
   for image in (ROOT/'slides/Co-optimize/output/B'/case['id']/stage).glob('*.png'):image.unlink()
  begin=time.perf_counter()
  with log.open('w')as stream:code=subprocess.call([str(ROOT/'.venv/bin/python'),'/tmp/cadgrasp_pso_post_worker.py',str(payload)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
  row=json.loads(result.read_text())if result.exists()else dict(id=case['id'],pose_count=case['pose_count'],experiment=exp,passed=False,status='RUNTIME_FAIL',error='exit '+str(code))
  row.update(process_wall_seconds=time.perf_counter()-begin,worker_exit_code=code,execution_log=str(log.relative_to(ROOT)));return row
 with ThreadPoolExecutor(max_workers=6)as pool:
  for future in as_completed([pool.submit(one,r)for r in cases if r['passed']]):
   row=future.result();rows.append(row);record();print('POST',policy,len(rows),'/52','PASS',sum(r['passed']for r in rows),row['id'],round(row['process_wall_seconds'],2),flush=True)
 # Failed groups have no previous success picture masquerading as a new one.
 for case in rows:
  if not case['passed']:
   for stage in ['step5.1','step5.2','step5.3','evaluation']:
    for image in (ROOT/'slides/Co-optimize/output/B'/case['id']/stage).glob('*.png'):image.unlink()
 print('POST POLICY COMPLETE',policy,flush=True)
protected=json.loads((SYSTEM/'protected_originals.json').read_text());changed=[p for p,h in protected.items()if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
save(SYSTEM/'postprocess_verification.json',dict(passed=not changed,changed=changed,protected_files=len(protected)))
assert not changed
print('BOTH POSTPROCESS CAMPAIGNS COMPLETE',flush=True)
