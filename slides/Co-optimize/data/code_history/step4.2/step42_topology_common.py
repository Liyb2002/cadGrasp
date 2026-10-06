"""Joint direction refinement guided by material separation."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from step42_topology import *

def solve(search):
    r=search.candidate(search.warm,label='joint topology warm',proxy=False)
    if r is not None:return r
    heights=np.sum(search.warm*search.normals,axis=1);centers=[]
    for k in np.argsort(-heights)[:3]:
        a=search.warm[k]-.01*search.normals[k];a/=np.linalg.norm(a)
        if not any(np.linalg.norm(a-c)<.01 for c in centers):centers.append(a)
    for center in centers:
        axis=np.eye(3)[np.argmin(abs(center))];u=np.cross(center,axis);u/=np.linalg.norm(u);v=np.cross(center,u)
        for degrees in [.5,1,2,5,10,20,40]:
            for theta in np.linspace(0,2*np.pi,24,endpoint=False):
                angle=np.deg2rad(degrees);a=np.cos(angle)*center+np.sin(angle)*(np.cos(theta)*u+np.sin(theta)*v)
                for lift in [.0001,.003,.01,.03]:
                    r=search.candidate(project_common(a,search.normals,lift),a,label='joint topology local tendency')
                    if r is not None:return r
    for count in [160,512,2048]:
        for a in fibonacci(count):
            r=search.candidate(project_common(a,search.normals),a,label='joint topology global tendency')
            if r is not None:return r
    return None
if __name__=='__main__':
    g=next(g for g in json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets'] if g['id']=='pose1+6+11+13+14+17');s=TopologySearch('B',g);began=time.monotonic();r=solve(s)
    if r is None:print('JOINT TOPOLOGY UNRESOLVED',flush=True)
    else:
        row=finish_connected(s,r,began);p=s.base/'step4/step4.2/data/report.json';report=json.loads(p.read_text());report['provenance']['code'].update(I.hashes([HERE/f for f in ['step4.2/step42_connected_safe.py','step4.2/step42_connected_local.py','step4.2/step42_connected_independent.py','step4.2/step42_contact_consistency.py','step4.2/step42_topology.py','step4.2/step42_topology_common.py']]));save(p,report);I.check_report(p);print(row,flush=True)
