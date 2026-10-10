from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import os,subprocess,time,sys
sys.path.insert(0,'slides/pose_set_search/code')
from case_sets import CASES
began=time.monotonic();env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
def run(case):
 start=time.monotonic()
 completed=subprocess.run(['.venv/bin/python','slides/pose_set_search/code/render_mesh_media.py','--case',case],env=env,capture_output=True,text=True)
 Path('/tmp/no_work_region_'+case+'.log').write_text(completed.stdout+completed.stderr)
 if completed.returncode:raise RuntimeError(case+': '+completed.stderr[-2500:])
 return case,round(time.monotonic()-start,1),completed.stdout.strip()
with ThreadPoolExecutor(max_workers=2) as pool:
 for future in as_completed([pool.submit(run,case) for case in CASES if case!='pose1-10']):
  case,seconds,output=future.result();print('COMPLETED',case,seconds,'seconds',output,flush=True)
print('NINE REMAINING CASES COMPLETE',round(time.monotonic()-began,1),'seconds',flush=True)
