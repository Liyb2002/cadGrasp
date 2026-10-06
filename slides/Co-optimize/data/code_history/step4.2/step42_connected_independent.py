"""Warm-start coordinate search of individual withdrawal directions."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from step42_connected_local import *
class IndependentConnectedSearch(LocalConnectedSearch):
    def connected_search(self):
        result=self.candidate(self.warm,label='warm independent paths',proxy=False)
        if result is not None:return result
        for round_index in range(8):
            previous=None if self.connected_best is None else self.connected_best['connection_score']
            for k in range(len(self.states)):
                basis=self.warm.copy() if self.connected_best is None else self.connected_best['directions'].copy()
                center=basis[k];normal=self.normals[k]
                tangent=center-(center@normal)*normal
                if np.linalg.norm(tangent)<1e-8:
                    axis=np.eye(3)[np.argmin(abs(normal))];tangent=axis-(axis@normal)*normal
                tangent/=np.linalg.norm(tangent);side=np.cross(normal,tangent)
                elevation=np.arcsin(np.clip(center@normal,-1,1))
                for degrees in [.1,-.1,.5,-.5,1,-1,2,-2,5,-5,10,-10,20,-20,40,-40,80,-80]:
                    for delta in [0.,.1,-.1,1.,-1.,5.,-5.,15.]:
                        az=np.deg2rad(degrees);el=np.clip(elevation+np.deg2rad(delta),1e-5,np.pi/2)
                        directions=basis.copy();directions[k]=np.cos(el)*(np.cos(az)*tangent+np.sin(az)*side)+np.sin(el)*normal
                        result=self.candidate(directions,label=f'individual connectivity pose {k} round {round_index}')
                        if result is not None:return result
                save(self.out/'data/independent_checkpoint.json',dict(proposals=self.evaluations,trace=self.connection_trace))
            if self.connected_best is None or previous==self.connected_best['connection_score']:break
        return super().connected_search()
def independent_case(name,group):
    search=IndependentConnectedSearch(name,group);began=time.monotonic();result=search.connected_search()
    if result is None:return dict(id=group['id'],passed=False)
    row=finish_connected(search,result,began);p=search.base/'step4/step4.2/data/report.json';r=json.loads(p.read_text());r['provenance']['code'].update(I.hashes([HERE/'step4.2/step42_connected_safe.py',HERE/'step4.2/step42_connected_local.py',HERE/'step4.2/step42_connected_independent.py']));save(p,r);I.check_report(p);return row
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--sets',nargs='+',default=['pose1+6+11+13+14+17','pose1+4+7+12+21+27']);args=parser.parse_args();groups=json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets'];groups=[g for g in groups if g['id'] in args.sets]
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:rows=list(pool.map(independent_case,['B']*len(groups),groups))
    save(HERE/'output/B/data/connected_independent_batch.json',dict(complete=all(r['passed'] for r in rows),results=rows));print('INDEPENDENT CONNECTED TOTAL',sum(r['passed'] for r in rows),'/',len(rows),flush=True)
