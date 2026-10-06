"""Redraw head figures from saved results without recomputing any candidates."""
import argparse,json,os,sys,tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'slides/baseline_algo'))
from codes.precompute_objects import head_cache as C
from codes.precompute_objects.precompute_heads import picture
from step3_scheculer.pair_tasks import read_task

def render(item):
    name,pose=item;folder=ROOT/'objects'/name/'poses'/pose;target=folder/'step2'
    cache=C.load(folder)
    if cache is None:raise FileNotFoundError(f'No completed head cache: {name}/{pose}')
    task=read_task(name,pose)
    geometry=SimpleNamespace(centers=cache['arrays']['centers'],scale=float(task.domain.mesh.extents.max()))
    fd,file=tempfile.mkstemp(prefix='.candidates-',suffix='.png',dir=target);os.close(fd);temporary=Path(file)
    try:
        picture(task,geometry,cache['pools'],temporary)
        temporary.replace(target/'candidates.png')
        report=cache['report'];report['artifacts']['candidates.png']=C.digest(target/'candidates.png')
        fd,metadata=tempfile.mkstemp(prefix='.report-',suffix='.json',dir=target);os.close(fd)
        staged=Path(metadata)
        try:
            staged.write_text(json.dumps(report,indent=2)+'\n')
            staged.replace(target/'report.json')
        finally:staged.unlink(missing_ok=True)
    finally:temporary.unlink(missing_ok=True)
    print('REDRAW',name,pose,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--object');p.add_argument('--pose');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if a.pose and not a.object:p.error('--pose requires --object')
    names=[a.object] if a.object else json.loads((ROOT/'objects/cases.json').read_text())['active_objects']
    jobs=[(name,a.pose or f'pose_{i}') for name in names for i in ([1] if a.pose else range(1,31))]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:list(pool.map(render,jobs))
if __name__=='__main__':main()
