import subprocess,json,time
from pathlib import Path
root=Path('slides/pose_set_search/output/B_work_cone_20261008');folder=root/'pose1+2+3+4+5+6+7+8+9+10';began=time.monotonic()
for script in ['mesh_media.py','render_mesh_media.py']:
 with (folder/('after_volume_polish_'+script+'.log')).open('w') as log:
  subprocess.run(['.venv/bin/python','slides/pose_set_search/code/'+script,'--root',str(root),'--case','pose1-10'],check=True,stdout=log,stderr=subprocess.STDOUT)
print('POLISHED TEN CHAIN MEDIA UPDATED',round(time.monotonic()-began,2),flush=True)
