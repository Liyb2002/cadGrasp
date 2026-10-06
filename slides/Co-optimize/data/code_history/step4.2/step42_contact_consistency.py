"""Exclude coplanar remnants incompatible with the initial exit velocity.

This injects a contact-extraction dependency into the two existing construction
functions, within a single worker process. It never modifies cached sources.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from step42_connected_independent import *
import step42 as force_module
import step42_connected as connected_module
from co_common import contact_boundary as original_boundary
_current_directions=None

def motion_consistent_boundary(mesh,shell,allowed_faces):
    tri,src=original_boundary(mesh,shell,allowed_faces)
    if _current_directions is not None and len(src):
        valid=np.max(mesh.face_normals[src]@_current_directions.T,axis=1)<=1e-9
        tri,src=tri[valid],src[valid]
    return tri,src

class ConsistentSearch(IndependentConnectedSearch):
    def candidate(self,directions,common=None,label='proposal',proxy=True):
        global _current_directions
        _current_directions=np.asarray(directions,float)
        force_module.contact_boundary=motion_consistent_boundary
        connected_module.contact_boundary=motion_consistent_boundary
        return super().candidate(directions,common,label,proxy)
    def connected_search(self):
        # Recover actual equilibrium first: an old coplanar-remnant pass is not
        # a feasible warm start after the contact correction.
        self.candidate(self.warm,label='contact-corrected previous paths',proxy=False)
        for round_index in range(8):
            before=None if self.best is None else self.best['score']
            for k in [5,2,1,0,4,3]:
                base=self.best['directions'].copy() if self.best is not None else self.warm.copy()
                n=self.normals[k];t=base[k]-(base[k]@n)*n;t/=np.linalg.norm(t);side=np.cross(n,t)
                for angle in [2,-2,5,-5,10,-10,20,-20,40,-40,80,-80,140,-140]:
                    for lift in [.0001,.001,.01,.05,.2]:
                        ds=base.copy();a=np.deg2rad(angle);ds[k]=np.cos(a)*t+np.sin(a)*side+lift*n;ds[k]/=np.linalg.norm(ds[k])
                        result=self.candidate(ds,label=f'contact-consistent critical pose {k}',proxy=False)
                        if result is not None:return result
                save(self.out/'data/consistent_checkpoint.json',dict(proposals=self.evaluations,best_counts=None if self.best is None else self.best['counts'],trace=self.connection_trace))
            if self.best is None or self.best['score']==before:break
        return super().connected_search()

def consistent_case(name,group):
    search=ConsistentSearch(name,group);began=time.monotonic();result=search.connected_search()
    if result is None:
        save(search.out/'data/consistent_unresolved.json',dict(complete=False,proposals=search.evaluations,best_counts=None if search.best is None else search.best['counts']));return dict(id=group['id'],passed=False)
    row=finish_connected(search,result,began);p=search.base/'step4/step4.2/data/report.json';r=json.loads(p.read_text());r['motion_consistent_contacts_required']=True;r['provenance']['code'].update(I.hashes([HERE/'step4.2/step42_connected_safe.py',HERE/'step4.2/step42_connected_local.py',HERE/'step4.2/step42_connected_independent.py',HERE/'step4.2/step42_contact_consistency.py']));save(p,r);I.check_report(p);return row
if __name__=='__main__':
    groups=json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets'];g=next(g for g in groups if g['id']=='pose1+4+7+12+21+27');print(consistent_case('B',g),flush=True)
