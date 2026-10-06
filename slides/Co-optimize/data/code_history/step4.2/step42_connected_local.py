"""Refine around feasible path tendencies instead of restarting a coarse sphere."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from step42_connected_safe import *
class LocalConnectedSearch(SafeConnectedSearch):
    def connected_search(self):
        result=self.candidate(self.warm,label='existing feasible paths',proxy=False)
        if result is not None:return result
        heights=np.sum(self.warm*self.normals,axis=1);lift=.0001 if heights.min()<.001 else .003 if heights.min()<.006 else .01
        centers=[]
        for k in np.argsort(-heights)[:3]:
            a=self.warm[k]-lift*self.normals[k];a/=np.linalg.norm(a)
            if not any(np.linalg.norm(a-c)<.01 for c in centers):centers.append(a)
        for center in centers:
            for base_lift in [.0001,.001,.003,.01,.03,.08]:
                result=self.candidate(project_common(center,self.normals,base_lift),center,label='warm inferred common tendency')
                if result is not None:return result
            axis=np.eye(3)[np.argmin(abs(center))];u=np.cross(center,axis);u/=np.linalg.norm(u);v=np.cross(center,u)
            for angle in [2,5,10,20,40,.5]:
                for theta in np.linspace(0,2*np.pi,32,endpoint=False):
                    a=np.cos(np.deg2rad(angle))*center+np.sin(np.deg2rad(angle))*(np.cos(theta)*u+np.sin(theta)*v)
                    for new_lift in [.0001,.001,.003,.01,.03]:
                        result=self.candidate(project_common(a,self.normals,new_lift),a,label='warm common local connectivity refinement')
                        if result is not None:return result
            save(self.out/'data/local_checkpoint.json',dict(proposals=self.evaluations,trace=self.connection_trace))
        return super().connected_search()

def local_case(name,group):
    search=LocalConnectedSearch(name,group);began=time.monotonic();result=search.connected_search()
    if result is None:
        save(search.out/'data/unresolved.json',dict(complete=False,proposals=search.evaluations,trace=search.connection_trace));return dict(id=group['id'],passed=False)
    row=finish_connected(search,result,began);p=search.base/'step4/step4.2/data/report.json';r=json.loads(p.read_text());r['provenance']['code'].update(I.hashes([HERE/'step4.2/step42_connected_safe.py',HERE/'step4.2/step42_connected_local.py']));save(p,r);I.check_report(p);return row
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--sets',nargs='+',default=['pose1+6+11+13+14+17','pose1+4+7+12+21+27']);args=parser.parse_args();groups=json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets'];groups=[g for g in groups if g['id'] in args.sets]
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:rows=list(pool.map(local_case,['B']*len(groups),groups))
    save(HERE/'output/B/data/connected_local_batch.json',dict(complete=all(r['passed'] for r in rows),results=rows));print('LOCAL CONNECTED TOTAL',sum(r['passed'] for r in rows),'/',len(rows),flush=True)
