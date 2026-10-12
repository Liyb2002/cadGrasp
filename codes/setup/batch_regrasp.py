"""Build complete per-object demo deliveries, with logs outside object folders."""
import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]


def run(name):
    lock=Path(tempfile.gettempdir())/f'cadgrasp-demo-batch-{name}.lock'
    with lock.open('w') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        return build(name)


def build(name):
    log=Path(tempfile.gettempdir())/f'cadgrasp-demo-batch-{name}.log'
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1')
    folder=ROOT/'objects'/name
    from verify_sequences import verify
    record=json.loads((folder/'poses.json').read_text())
    searched=record.get('trajectory_revision')=='diverse_regrasp_v1'
    if searched and (folder/'video.mp4').exists():
        verify(folder)
        shutil.copy2(folder/'video.mp4',ROOT/'simulation/videos'/f'{name}_grounded_sequence.mp4')
        return dict(object=name,complete=True,already_complete=True,video=str(folder/'video.mp4'))
    with log.open('a') as out:
        for pairs in (() if searched else (240,480)):
            command=[sys.executable,str(ROOT/'codes/setup/demo_regrasp.py'),name,
                     '--pairs',str(pairs),'--candidate-budget','160','--downward-component','.20']
            out.write('\n'+json.dumps(dict(command=command))+'\n');out.flush()
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT)
            if result.returncode==0:break
        if not searched and result.returncode:
            return dict(object=name,complete=False,stage='search',log=str(log))
        command=[sys.executable,str(ROOT/'codes/setup/sequence.py'),name,'--replay']
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT)
        if result.returncode:return dict(object=name,complete=False,stage='video',log=str(log))
    verify(folder)
    shutil.copy2(folder/'video.mp4',ROOT/'simulation/videos'/f'{name}_grounded_sequence.mp4')
    return dict(object=name,complete=True,video=str(folder/'video.mp4'),log=str(log))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects',nargs='+');parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(run,name):name for name in args.objects}
        for future in as_completed(pending):
            try:result=future.result()
            except Exception as error:result=dict(object=pending[future],complete=False,error=str(error))
            print(json.dumps(result),flush=True)
