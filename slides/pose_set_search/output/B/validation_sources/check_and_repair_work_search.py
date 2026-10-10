import sys,json,time,subprocess,os,hashlib
from pathlib import Path
sys.path.insert(0,'slides/pose_set_search/code')
from case_sets import CASES,case_directory
root=Path('slides/pose_set_search/output/B_work_cone_20261008');completed={};began=time.monotonic()
while len(completed)<20:
 progress=False
 for case in CASES:
  for method in ['whole','incremental']:
   key=case+'/'+method
   if key in completed:continue
   folder=case_directory(root,case)/method;p=folder/'search_report.json'
   if not p.exists() or not (folder/'sampled_layout.npz').exists():continue
   r=json.loads(p.read_text())
   command=['.venv/bin/python','slides/pose_set_search/code/repair_fast_search.py','--root',str(root),'--case',case,'--method',method]
   if r['sampled_force_passed'] and r['pose_count']==r['requested_pose_count']:command.append('--check-only')
   print('CHECK OR REPAIR',key,flush=True)
   with (folder/'strict_recheck.log').open('w') as log:
    subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=750,
      env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
   report=json.loads(p.read_text())
   check=json.loads((folder/'strict_sampled_recheck.json').read_text())
   complete=report['sampled_force_passed'] and report['pose_count']==report['requested_pose_count']
   if not complete:raise RuntimeError('Local continuation unresolved: '+key)
   if not check['sampled_force_passed']:
    with (folder/'strict_recheck_after_repair.log').open('w') as log:
     subprocess.run(command+['--check-only'],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=120)
    check=json.loads((folder/'strict_sampled_recheck.json').read_text())
   assert check['sampled_force_passed'] and check['full_requested_set_evaluated']
   completed[key]=dict(passed=True,counts=check['counts'],sampled_layout_sha256=check['sampled_layout_sha256'])
   (root/'strict_sampled_batch.json').write_text(json.dumps(dict(complete=len(completed)==20,results=completed,pressure_acceptance=False,wall_seconds=time.monotonic()-began),indent=2)+'\n')
   print('STRICT PASSED',key,flush=True);progress=True
 if not progress:time.sleep(2)
print('ALL20 STRICT SAMPLED REPLAYS PASSED',flush=True)
