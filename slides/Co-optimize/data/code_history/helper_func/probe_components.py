import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from step42_timed import *
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def probe(g):
    search=Search('B',g);out=search.out;source=trimesh.load(out/'remaining_support.obj',force='mesh',process=False);parts=S.solid(source).decompose();parts.sort(key=lambda p:-material_volume(p));rows=[]
    for index,part in enumerate(parts):
        tri,src=contact_boundary(search.mesh,S.unpack(part),search.allowed)
        if not len(tri):continue
        counts=[]
        signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,15.)
        try:
            for task,T in search.states:counts.append(int(J.classify(supply(task,T,tri,src),task.targets)[0].sum()))
        except EvaluationDeadline:counts.append(-1)
        finally:signal.setitimer(signal.ITIMER_REAL,0.)
        rows.append(dict(component=index,volume_cm3=material_volume(part)*1e6,counts=counts))
        if len(counts)==len(g['poses']) and all(c==32768 for c in counts):break
    record=dict(id=g['id'],component_count=len(parts),results=rows);save(out/'data/component_probe.json',record);print(record,flush=True);return record
if __name__=='__main__':
 groups=json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets']
 with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:rows=list(pool.map(probe,groups))
 save(HERE/'output/B/data/component_probe.json',rows)
