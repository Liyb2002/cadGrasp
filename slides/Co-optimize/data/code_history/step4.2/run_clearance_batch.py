"""Fresh 1% clearance construction/optimization for normal and illegal B sets.

Runs actual construction acceptance only; never creates public images or videos.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from co_common import *
import argparse, time, traceback, multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from step41 import prepare, run as initialize
from step42_dispatch import dispatch

def groups_for(scope):
    groups=[]
    if scope in ('all','legal'):
        groups+=json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets']
    if scope in ('all','illegal'):
        groups += [dict(g,id='illegal/'+g['id']) for g in json.loads((ROOT/'objects/B/illegal_pose_sets.json').read_text())['sets']]
    return groups

def case(group,initialization):
    began=time.monotonic();base=HERE/'output/B'/group['id'];record=dict(id=group['id'],passed=False,status='running')
    try:
        record['initialization']=initialize('B',group,initialization,render_images=False)
        record.update(dispatch('B',group))
        record['status']='force_exit_connectivity_pass' if record['passed'] else 'bounded_search_unresolved'
        current=base/'step4/step4.2/data/report.json'
        if record['passed']:
            report=I.check_report(current)
            assert report['exit_clearance']['fraction_of_object_max_extent']==.01
            assert report['clearance_diagnostics']['contact_check_performed']
            record['state_results']=report['state_results']
            record['exit_clearance']=report['exit_clearance']
            record['remaining_component_count']=report['remaining_component_count']
        else:
            record['state_results']=[]
    except Exception as error:
        record.update(status='execution_unresolved',error=str(error),traceback=traceback.format_exc())
        traceback.print_exc()
    record['seconds']=time.monotonic()-began
    save(base/'step4/step4.2/data/clearance_rerun.json',record)
    print('CLEARANCE CASE',group['id'],record['status'],flush=True)
    return record

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope',choices=['all','legal','illegal'],default='all')
    parser.add_argument('--sets',nargs='+');parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--pending-from',type=Path,help='Exclude completed groups in this existing batch record; do not reuse their acceptance')
    parser.add_argument('--summary-name',default='step42_clearance_batch.json',help='Independent batch filename under output/B/data')
    args=parser.parse_args();groups=groups_for(args.scope)
    if args.sets:groups=[g for g in groups if g['id'] in args.sets]
    if args.pending_from:
        completed={r['id'] for r in json.loads(args.pending_from.read_text())['results']}
        groups=[g for g in groups if g['id'] not in completed]
    initialization=prepare('B');rows=[]
    target=HERE/'output/B/data'/args.summary_name
    def checkpoint():
        ordered=sorted(rows,key=lambda r:next(i for i,g in enumerate(groups) if g['id']==r['id']))
        batch=dict(complete=len(rows)==len(groups),scope=args.scope,sets=len(groups),completed_sets=len(rows),passed_sets=sum(r['passed'] for r in rows),unresolved_sets=sum(not r['passed'] for r in rows),not_run=[g['id'] for g in groups if g['id'] not in {r['id'] for r in rows}],pose_instances=sum(len(g['poses']) for g in groups),results=ordered,policy='Fresh 1% clearance; all original loads, full continuous exits, one connected component; no video, static rendering or export replay',provenance=provenance([ROOT/'objects/B/pose_sets.json',ROOT/'objects/B/illegal_pose_sets.json'],[HERE/'step4.2/run_clearance_batch.py',HERE/'helper_func/exit_clearance.py']))
        save(target,batch)
        if batch['complete'] and args.summary_name=='step42_clearance_batch.json':save(HERE/'output/B/data/step42_batch.json',batch)
    checkpoint()
    with ProcessPoolExecutor(max_workers=args.jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(case,g,initialization):g for g in groups}
        for future in as_completed(futures):
            rows.append(future.result());checkpoint()
    print('CURRENT CLEARANCE TOTAL',sum(r['passed'] for r in rows),'/',len(rows),flush=True)

if __name__=='__main__':main()
