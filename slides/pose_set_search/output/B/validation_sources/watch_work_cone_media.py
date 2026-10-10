import os,sys,time,json,subprocess
from pathlib import Path
sys.path.insert(0,'slides/pose_set_search/code')
from case_sets import CASES,case_directory
root=Path('slides/pose_set_search/output/B_work_cone_20261008')
ledger=root/'media_batch.json'
prior=json.loads(ledger.read_text()) if ledger.exists() else {}
completed=prior.get('results',{});previous_wall=prior.get('wall_seconds',0);began=time.monotonic()
while len(completed)<len(CASES):
 progress=False
 for case in CASES:
  if case in completed:continue
  folder=case_directory(root,case)
  ready=True
  for m in ['whole','incremental']:
   path=folder/m/'search_report.json';check=folder/m/'strict_sampled_recheck.json'
   if not path.exists() or not check.exists():ready=False;break
   report=json.loads(path.read_text());verified=json.loads(check.read_text())
   if not report['sampled_force_passed'] or report['pose_count']!=report['requested_pose_count'] or not verified['sampled_force_passed']:ready=False;break
  if not ready:continue
  stage=time.monotonic();print('MEDIA START',case,flush=True)
  with (folder/'mesh_media.log').open('w') as log:
   subprocess.run(['.venv/bin/python','slides/pose_set_search/code/mesh_media.py','--case',case,'--root',str(root)],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=300)
  with (folder/'media.log').open('w') as log:
   subprocess.run(['.venv/bin/python','slides/pose_set_search/code/render_mesh_media.py','--case',case,'--root',str(root)],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=400)
  completed[case]=dict(complete=True,seconds=time.monotonic()-stage,directory=folder.name)
  (root/'media_batch.json').write_text(json.dumps(dict(complete=len(completed)==10,results=completed,wall_seconds=previous_wall+time.monotonic()-began,previous_pipeline_wall_seconds=previous_wall,resumed_pipeline_wall_seconds=time.monotonic()-began),indent=2)+'\n')
  print('MEDIA DONE',case,completed[case]['seconds'],flush=True);progress=True
 if not progress:time.sleep(5)
print('ALL TEN WORK-CONE MEDIA SETS COMPLETE',flush=True)
