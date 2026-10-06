"""Resume validated recovery outputs or run bounded path recovery for all sets."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from step42_dispatch import *
import multiprocessing,shutil
from concurrent.futures import ProcessPoolExecutor

def run_case(name,group,force=False):
    target=(HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name)/group['id']/'step4/step4.2'
    if not force:
        try:
            r=I.check_report(target/'data/report.json')
            if r.get('exit_clearance',{}).get('fraction_of_object_max_extent')!=.01:raise ValueError('Historical result has no mandatory 1% exit clearance')
            if r['passed'] and r['connectivity_required'] and r['remaining_component_count']==1:
                return dict(id=group['id'],passed=True,connected=True,component_count=1,proposals=r['proposal_count'],restored_volume_cm3=r['restored_volume_cm3'],pruned_volume_cm3=r['pruned_unnecessary_volume_cm3'],seconds=r['seconds'],reused_validated_record=True)
        except (AssertionError,FileNotFoundError,KeyError,ValueError,RuntimeError):pass
    return dispatch(name,group)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--sets',nargs='+');parser.add_argument('--jobs',type=int,default=2);parser.add_argument('--fresh',action='store_true',help='Reconstruct/reclassify using previous feasible directions as warm starts');args=parser.parse_args();manifest=ROOT/'objects/B/pose_sets.json';groups=json.loads(manifest.read_text())['sets'];groups=[g for g in groups if not args.sets or g['id'] in args.sets]
    with ProcessPoolExecutor(max_workers=args.jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(run_case,'B',g,args.fresh) for g in groups];rows=[f.result() for f in futures]
    reports=[HERE/'output/B'/g['id']/'step4/step4.2/data/report.json' for g,r in zip(groups,rows) if r['passed']]
    batch=dict(complete=all(r['passed'] for r in rows),sets=len(rows),passed_sets=sum(r['passed'] for r in rows),pose_instances=sum(len(g['poses']) for g in groups),results=rows,provenance=provenance([manifest]+reports,[HERE/'step4.2/step42_connected_all.py']))
    save(HERE/'output/B/data/step42_batch.json',batch);print('STEP4.2',batch['passed_sets'],'/',len(rows),'sets; all original loads per pose; one connected support required',flush=True)
    if not batch['complete']:raise SystemExit(2)
if __name__=='__main__':main()
