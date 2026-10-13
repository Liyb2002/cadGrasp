"""Recover whole-batch summaries from fingerprinted case reports, no search."""
import _bootstrap
from collections import Counter
import argparse,time
from co_common import *
from run_all import saved_groups,output_root
from whole_pipeline import publish,SCHEMA


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object',nargs='?',default='B');args=parser.parse_args()
    root=output_root(args.object);rows=[]
    for group in saved_groups(args.object):
        base=root/group['id']/'step4';path=base/'data/report.json'
        row=None
        if path.exists():
            try:
                saved=json.loads(path.read_text())
                if saved.get('schema')==SCHEMA and saved.get('passed'):
                    initial=I.check_report(base/'step4.1/data/report.json')
                    actual=I.check_report(base/'step4.2/data/report.json')
                    assert actual['force_exit_work_passed'] and actual['poses']==group['poses']
                    assert all(n==32768 for n in actual['counts'].values())
                    assert initial['source_initialization_schema']=='whole_registered_work_cones_v1'
                    row=saved
            except (OSError,ValueError,RuntimeError,AssertionError,KeyError):pass
        if row is None:
            row=dict(id=group['id'],poses=group['poses'],status='pending',passed=False)
            failure=base/'data/failure.json';marker=base/'data/run_schema.json'
            if failure.exists() and marker.exists() and failure.stat().st_mtime>=marker.stat().st_mtime:
                try:row=json.loads(failure.read_text())
                except ValueError:pass
        rows.append(row)
    statuses=dict(Counter(row['status'] for row in rows))
    complete=all(row['status'] in ['pass','unresolved'] for row in rows)
    previous=root/'data/whole_step4_batch.json'
    prior=json.loads(previous.read_text()) if previous.exists() else {}
    # A controller may have run only a small chunk. Its options and wall time
    # must not be presented as the budget/time of the recovered full batch.
    controllers=list(prior.get('controller_batches',[]))
    if not prior.get('summary_recovered_from_current_case_reports') and prior.get('options'):
        controller={key:prior[key] for key in ['sets','options','seconds','passed','unresolved'] if key in prior}
        if controller not in controllers:controllers.append(controller)
    batch=dict(controller_batches=controllers,
        case_seconds_sum=sum(row.get('seconds',0.) for row in rows),
        case_seconds_sum_is_not_parallel_wall_time=True)
    batch.update(complete=complete,schema=SCHEMA,object=args.object,stage='both',sets=len(rows),
        passed=sum(row['passed'] for row in rows),unresolved=sum(row['status']=='unresolved' for row in rows),
        pending=sum(row['status']=='pending' for row in rows),status_counts=statuses,results=rows,
        full_fixture_accepted=False,summary_recovered_from_current_case_reports=True,
        collector_provenance=provenance([root/'data/whole_step4_sources/manifest.json'],[Path(__file__)]))
    save(root/'data/whole_step4_progress.json',rows);save(previous,batch)
    publish(args.object,rows,'both')
    print('COLLECTED',batch['passed'],'/',len(rows),'pass',batch['unresolved'],'unresolved',batch['pending'],'pending',flush=True)


if __name__=='__main__':main()
