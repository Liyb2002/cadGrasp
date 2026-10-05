"""Target a path identified by leave-one-path-out restoration diagnostics."""
from step42_refine import *
if __name__=='__main__':
    g=next(g for g in json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets'] if g['id']=='pose1+4+7+12+21+27');began=time.monotonic();search=RefineSearch('B',g)
    saved=search.out/'data/best_paths.npz';directions=np.load(saved)['directions'].copy();search.out=search.base/'step4/step4.2/data/local_run';(search.out/'data').mkdir(parents=True,exist_ok=True)
    result=search.candidate(directions,label='independent warm baseline',proxy=False)
    for round_index in range(4):
        if result is not None:break
        for k in [2,5,1,0,4,3]:
            n=search.normals[k];center=search.best['directions'][k].copy() if search.best is not None else directions[k];tangent=center-(center@n)*n;tangent/=np.linalg.norm(tangent);side=np.cross(n,tangent)
            for angle in [1,-1,2,-2,5,-5,10,-10,20,-20,40,-40,80,-80,140,-140]:
                for lift in [.001,.01,.05,.2]:
                    candidate=search.best['directions'].copy() if search.best is not None else directions.copy();theta=np.deg2rad(angle);candidate[k]=np.cos(theta)*tangent+np.sin(theta)*side+lift*n;candidate[k]/=np.linalg.norm(candidate[k])
                    result=search.candidate(candidate,label=f'independent critical pose {k} angle {angle} lift {lift}',proxy=False)
                    if result is not None:break
                if result is not None:break
            if result is not None:break
    if result is None:
        save(search.out/'data/unresolved.json',dict(best_counts=None if search.best is None else search.best['counts'],proposals=search.evaluations,timeouts=search.timeouts));print('LOCAL UNRESOLVED',flush=True);raise SystemExit(2)
    row=finish(search,result,began);p=search.out/'data/report.json';r=json.loads(p.read_text());r['provenance']['code'].update(I.hashes([HERE/'step42_refine.py',HERE/'step42_local.py']));r['candidate_timeouts_unresolved']=search.timeouts;r['search_method']+='; independent critical-path refinement from leave-one-path-out diagnostics';save(p,r);I.check_report(p)
    target=search.base/'step4/step4.2'
    for f in search.out.iterdir():
        if f.is_dir():
            for item in f.iterdir():shutil.copy2(item,target/'data'/item.name)
        else:shutil.copy2(f,target/f.name)
    I.check_report(target/'data/report.json');print('LOCAL PASS',row,flush=True)
