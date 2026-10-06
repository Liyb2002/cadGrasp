"""Balance the worst pose rather than locking in five complete poses."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from step42_contact_consistency import *
class BalancedSearch(ConsistentSearch):
    def candidate(self,directions,common=None,label='proposal',proxy=True):
        old=len(self.trace);r=super().candidate(directions,common,label,proxy)
        if len(self.trace)>old:
            row=self.trace[-1]
            if 'actual_covered_counts' in row:
                counts=row['actual_covered_counts'];score=(min(counts),sum(counts),sum(c==32768 for c in counts))
                if getattr(self,'balanced_best',None) is None or score>self.balanced_best['balanced_score']:
                    ds=np.asarray(directions,float);ds=ds/np.linalg.norm(ds,axis=1)[:,None]
                    self.balanced_best=dict(directions=ds.copy(),counts=counts,balanced_score=score,score=(int(sum(c==32768 for c in counts)),min(counts)/32768,sum(counts)/32768))
                    np.savez(self.out/'data/balanced_best_paths.npz',directions=ds,counts=counts);print('BALANCED RECOVERY',self.group['id'],counts,'proposal',self.evaluations,flush=True)
        if getattr(self,'balanced_best',None) is not None:self.best=self.balanced_best.copy()
        return r
if __name__=='__main__':
    g=next(g for g in json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets'] if g['id']=='pose1+4+7+12+21+27');s=BalancedSearch('B',g)
    # Start from the strongest balanced candidate of the preceding coordinate
    # sweep (original warm direction, pose 7 azimuth -5 degrees).
    k=2;n=s.normals[k];t=s.warm[k]-(s.warm[k]@n)*n;t/=np.linalg.norm(t);side=np.cross(n,t);a=np.deg2rad(-5);s.warm[k]=np.cos(a)*t+np.sin(a)*side+.0001*n;s.warm[k]/=np.linalg.norm(s.warm[k]);np.savez(s.out/'data/warm_paths.npz',directions=s.warm)
    began=time.monotonic();r=s.connected_search()
    if r is None:print('BALANCED UNRESOLVED',flush=True)
    else:
        row=finish_connected(s,r,began);p=s.base/'step4/step4.2/data/report.json';report=json.loads(p.read_text());report['motion_consistent_contacts_required']=True;report['provenance']['code'].update(I.hashes([HERE/f for f in ['step4.2/step42_connected_safe.py','step4.2/step42_connected_local.py','step4.2/step42_connected_independent.py','step4.2/step42_contact_consistency.py','step4.2/step42_balanced.py']]));save(p,report);I.check_report(p);print(row,flush=True)
