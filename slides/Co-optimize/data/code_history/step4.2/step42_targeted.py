"""Independent critical-path search after shared-tendency recovery."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from step42_timed import *
import shutil

def refine(search,directions,order):
    result=search.candidate(directions,label='independent warm baseline',proxy=False)
    for round_index in range(5):
        if result is not None:break
        before=None if search.best is None else search.best['score']
        for k in order:
            n=search.normals[k];center=search.best['directions'][k].copy() if search.best is not None else directions[k];tangent=center-(center@n)*n
            if np.linalg.norm(tangent)<1e-10:
                axis=np.eye(3)[np.argmin(abs(n))];tangent=axis-(axis@n)*n
            tangent/=np.linalg.norm(tangent);side=np.cross(n,tangent)
            for angle in [1,-1,2,-2,5,-5,10,-10,20,-20,40,-40,80,-80,140,-140]:
                for lift in [.001,.01,.05,.2]:
                    candidate=search.best['directions'].copy() if search.best is not None else directions.copy();theta=np.deg2rad(angle);candidate[k]=np.cos(theta)*tangent+np.sin(theta)*side+lift*n;candidate[k]/=np.linalg.norm(candidate[k])
                    result=search.candidate(candidate,label=f'independent critical pose {k} angle {angle} lift {lift}',proxy=False)
                    if result is not None:break
                if result is not None:break
            if result is not None:break
        if search.best is not None and search.best['score']==before:break
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--set',default='pose1+4+7+12+21+27');parser.add_argument('--warm',default='data/local_run/data/best_paths.npz');args=parser.parse_args()
    g=next(g for g in json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets'] if g['id']==args.set);began=time.monotonic();search=TimeLimitedSearch('B',g);warm_path=search.out/args.warm;directions=np.load(warm_path)['directions'].copy();search.out=search.base/'step4/step4.2/data/targeted_run';(search.out/'data').mkdir(parents=True,exist_ok=True);np.savez(search.out/'data/warm_start.npz',directions=directions)
    result=refine(search,directions,[5,2,1,0,4,3] if len(g['poses'])==6 else list(range(len(g['poses']))))
    if result is None:
        save(search.out/'data/unresolved.json',dict(best_counts=None if search.best is None else search.best['counts'],proposals=search.evaluations,timeouts=search.timeouts));print('TARGETED UNRESOLVED',flush=True);raise SystemExit(2)
    row=finish(search,result,began);p=search.out/'data/report.json';r=json.loads(p.read_text());r['provenance']['code'].update(I.hashes([HERE/'step4.2/step42_timed.py',HERE/'step4.2/step42_targeted.py']));r['provenance']['inputs'].update(I.hashes([search.out/'data/warm_start.npz']));r['candidate_timeouts_unresolved']=search.timeouts;r['search_method']+='; independent critical-path refinement from saved warm directions';save(p,r);I.check_report(p)
    target=search.base/'step4/step4.2'
    for f in search.out.iterdir():
        if f.is_dir():
            for item in f.iterdir():shutil.copy2(item,target/'data'/item.name)
        else:shutil.copy2(f,target/f.name)
    I.check_report(target/'data/report.json');print('TARGETED PASS',row,flush=True)
