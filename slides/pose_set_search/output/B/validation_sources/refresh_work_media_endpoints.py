import json,shutil,subprocess,time
from pathlib import Path
root=Path('slides/pose_set_search/output/B_work_cone_20261008')
for case in ['seven_chain','seven_hard']:
 import sys
 sys.path.insert(0,'slides/pose_set_search/code')
 from case_sets import case_directory
 folder=case_directory(root,case);data=json.loads((folder/'videos.json').read_text())
 if data['process'].get('process_animation_uses_saved_endpoint_poses'):continue
 history=folder/'_history/before_endpoint_animation';history.mkdir(parents=True,exist_ok=True)
 for filename in ['process.mp4','result.mp4','videos.json']:
  shutil.copy2(folder/filename,history/filename)
 began=time.monotonic()
 with (folder/'endpoint_media_refresh.log').open('w') as log:
  subprocess.run(['.venv/bin/python','slides/pose_set_search/code/render_mesh_media.py','--root',str(root),'--case',case],stdout=log,stderr=subprocess.STDOUT,check=True)
 print('REFRESHED',case,round(time.monotonic()-began,1),flush=True)
