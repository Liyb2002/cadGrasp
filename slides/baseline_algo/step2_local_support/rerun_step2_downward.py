"""Recompute Step 2 in each existing object/pose output directory."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from step3_scheculer import contacts as I

HERE=Path(__file__).resolve().parent.parent
OUT=HERE/'output'
CASES=[(name,f'pose_{i}') for name in ('A1-f','B','C5') for i in range(1,5)]
STAGES=['circles.py','insertion_directions.py','audit.py','sampling_audit.py','draw.py']


def run_case(name,pose):
    root=OUT/name/pose/'step2_local_support'
    root.mkdir(parents=True,exist_ok=True)
    inputs=I.hashes(list((root.parent/'step_1_needs').iterdir()))
    record=dict(object=name,pose=pose,complete=False,started_at=datetime.now(timezone.utc).isoformat(),
                scope='Step 2 only, overwrite in place',step1_inputs=inputs)
    I.save(root/'run.json',record)
    env=dict(os.environ,CADGRASP_POSE=pose,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    start=time.monotonic();code=0
    with (root/'run.log').open('w') as stream:
        for stage in STAGES:
            print(f'[{name} / {pose}] {stage}',file=stream,flush=True)
            command=[sys.executable,str(HERE/'step2_local_support'/stage),name]
            if stage=='circles.py':command.append('--reuse-work-volume')
            code=subprocess.run(command,env=env,cwd=HERE.parents[1],stdout=stream,stderr=subprocess.STDOUT).returncode
            if code:break
    I.check_hashes(inputs)
    record.update(complete=code==0,returncode=code,elapsed_seconds=time.monotonic()-start,
                  finished_at=datetime.now(timezone.utc).isoformat())
    I.save(root/'run.json',record)
    return dict(object=name,pose=pose,returncode=code,elapsed_seconds=record['elapsed_seconds'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    # Validate unchanged access geometry before overwriting any candidate output.
    for name,pose in CASES:
        root=OUT/name/pose/'step2_local_support'
        I.check_report(root/'work_volume.json')
        assert I.check_report(root/'work_volume_audit.json')['passed']
    records=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending=[pool.submit(run_case,*case) for case in CASES]
        for future in as_completed(pending):
            row=future.result();records.append(row);print(json.dumps(row),flush=True)
    if any(r['returncode'] for r in records):return 1
    from step2_local_support.summary_downward import build
    build()
    return 0


if __name__=='__main__':sys.exit(main())
