"""One-time exact sampling replay for caches predating the finite-center repair.

Only replay-identical caches may retain their existing geometry/certificates.
Nonfinite or changed centers require full recomputation by precompute_heads.py.
"""
import argparse,json,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'slides/DSL_algo'))
import numpy as np
from codes.precompute_objects import head_cache as C
from codes.precompute_objects.head_geometry import PairGeometry,FreePaths
from step3_scheculer.pair_tasks import read_task

def replay(item):
    path,previous=item;folder=Path(path).parent.parent;target=folder/'step2';report=json.loads(Path(path).read_text())
    current=C.sources()
    if report['sources']==current:return dict(object=report['object'],pose=report['pose'],status='current')
    if report['sources']!=previous:return dict(object=report['object'],pose=report['pose'],status='rebuild_other_revision')
    changed=[k for k in current if previous.get(k)!=current[k]]
    if changed!=['codes/precompute_objects/heads/surface.py']:raise ValueError(f'Not solely the sampling repair: {changed}')
    if report['inputs']!=C.inputs(folder):raise ValueError('Changed inputs during cache migration')
    for name,value in report['artifacts'].items():
        if C.digest(target/name)!=value:raise ValueError('Corrupt cache artifact')
    with np.load(target/'geometry.npz') as z:centers=z['centers'].copy();faces=z['center_faces'].copy()
    if not np.isfinite(centers).all():return dict(object=report['object'],pose=report['pose'],status='rebuild_nonfinite')
    task=read_task(report['object'],report['pose'])
    with patch.object(FreePaths,'__init__',return_value=None):
        g=PairGeometry([task],count=report['count'],initialize_candidates=False,use_precomputed=False)
    if not (np.array_equal(centers,g.centers) and np.array_equal(faces,g.faces)):
        return dict(object=report['object'],pose=report['pose'],status='rebuild_changed_centers')
    report['previous_sources']=previous;report['sources']=current
    report['cache_migration']=dict(reason='Finite-center sampler repair',acceptance='Exact replay of all centers and source-face IDs; all other computation sources and numerical artifacts unchanged',
        replay_script_sha256=C.digest(__file__),recomputed_candidates=False)
    staged=target/'.replayed-report.json';staged.write_text(json.dumps(report,indent=2)+'\n');staged.replace(Path(path))
    return dict(object=report['object'],pose=report['pose'],status='reused_exact_sampling_replay')

def main():
    p=argparse.ArgumentParser();p.add_argument('previous_sources');p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    previous=json.loads(Path(a.previous_sources).read_text());jobs=[(str(path),previous) for path in (ROOT/'objects').glob('*/poses/pose_*/step2/report.json')]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(replay,jobs))
    (ROOT/'codes/precompute_objects/finite_center_migration.json').write_text(json.dumps(dict(cases=rows),indent=2)+'\n')
    from collections import Counter
    print(json.dumps(dict(Counter(r['status'] for r in rows))))
if __name__=='__main__':main()
