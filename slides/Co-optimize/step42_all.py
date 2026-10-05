"""Resume validated recovery outputs or run bounded path recovery for all sets."""
from step42_timed import *
from step42_targeted import refine
import multiprocessing,shutil
from concurrent.futures import ProcessPoolExecutor

def run_case(name,group,force=False):
    target=HERE/'output'/name/group['id']/'step4/step4.2';existing=None
    if (target/'data/report.json').exists():
        try:
            existing=I.check_report(target/'data/report.json')
            if not existing['passed']:existing=None
        except (AssertionError,FileNotFoundError):existing=None
    if existing is not None and not force:
        return dict(id=group['id'],passed=True,proposals=existing['proposal_count'],restored_volume_cm3=existing['restored_volume_cm3'],seconds=existing['seconds'],reused_validated_record=True)
    began=time.monotonic();search=TimeLimitedSearch(name,group);search.out=target/'data/current_run';(search.out/'data').mkdir(parents=True,exist_ok=True)
    result=None;warm=None
    if existing is not None:
        warm=np.array([row['direction_fixture'] for row in existing['state_results']]);np.savez(search.out/'data/warm_start.npz',directions=warm)
        result=search.candidate(warm,label='independent previous feasible paths',proxy=False)
    if result is None:result=search.search()
    if result is None and search.best is not None:
        result=refine(search,search.best['directions'],list(range(len(group['poses']))))
    if result is None:
        save(search.out/'data/unresolved.json',dict(complete=False,proposals=search.evaluations,best_counts=None if search.best is None else search.best['counts']))
        return dict(id=group['id'],passed=False,status='search_unresolved')
    row=finish(search,result,began);p=search.out/'data/report.json';r=json.loads(p.read_text());r['provenance']['code'].update(I.hashes([HERE/'step42_all.py',HERE/'step42_timed.py',HERE/'step42_targeted.py']));r['candidate_timeouts_unresolved']=search.timeouts
    if warm is not None:r['provenance']['inputs'].update(I.hashes([search.out/'data/warm_start.npz']))
    save(p,r)
    for f in search.out.iterdir():
        if f.is_dir():
            for item in f.iterdir():shutil.copy2(item,target/'data'/item.name)
        else:shutil.copy2(f,target/f.name)
    I.check_report(target/'data/report.json');return row

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--sets',nargs='+');parser.add_argument('--jobs',type=int,default=2);parser.add_argument('--fresh',action='store_true',help='Reconstruct/reclassify using previous feasible directions as warm starts');args=parser.parse_args();manifest=ROOT/'objects/B/pose_sets.json';groups=json.loads(manifest.read_text())['sets'];groups=[g for g in groups if not args.sets or g['id'] in args.sets]
    with ProcessPoolExecutor(max_workers=args.jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(run_case,'B',g,args.fresh) for g in groups];rows=[f.result() for f in futures]
    reports=[HERE/'output/B'/g['id']/'step4/step4.2/data/report.json' for g,r in zip(groups,rows) if r['passed']]
    batch=dict(complete=all(r['passed'] for r in rows),sets=len(rows),passed_sets=sum(r['passed'] for r in rows),pose_instances=sum(len(g['poses']) for g in groups),results=rows,provenance=provenance([manifest]+reports,[HERE/'step42_all.py']))
    save(HERE/'output/B/data/step42_batch.json',batch);print('STEP4.2',batch['passed_sets'],'/',len(rows),'sets; all original loads per pose; connectivity deferred',flush=True)
    if not batch['complete']:raise SystemExit(2)
if __name__=='__main__':main()
