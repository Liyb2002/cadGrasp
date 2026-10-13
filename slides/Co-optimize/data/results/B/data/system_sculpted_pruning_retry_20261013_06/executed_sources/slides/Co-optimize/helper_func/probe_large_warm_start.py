"""Recheck a preserved OLD WHOLE layout with NEW actual mesh/force physics.

This is a labelled warm-start experiment, never an independent cold-start result.
It does not use incremental or reuse old forces/support geometry as certificates.
"""
import _bootstrap
import argparse
import os
import time
from co_common import *
from run_all import output_root
from run_large_pose_sets import groups,OLD
from whole_pipeline import load_layout
from whole_search.fast_search import FastModel
from whole_search.search import failed_count
from whole_search.common import legal_direction
from whole_search.exact_worker import write_result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('case');args=parser.parse_args()
    group=next(g for g in groups() if args.case in [g['id'],g['source_case']])
    old=OLD/'output/B'/group['id']/'whole'
    old_report=json.loads((old/'search_report.json').read_text())
    assert old_report['poses']==group['poses'] and old_report['sampled_force_passed']
    base=output_root('B')/group['id'];out=base/'step4/step4.2/old_whole_warm_probes'/time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True);source=old/'sampled_layout.npz';layout=load_layout(source)
    model=FastModel(group['poses'],'B',initialization_report=base/'step3/step3.1/data/report.json')
    os.environ['COOPT_WHOLE_RUN_TOKEN']=json.loads((output_root('B')/'data/whole_step4_active_run.json').read_text())['run_token']
    model.worker_program=HERE/'helper_func/whole_search/exact_worker.py';model.checkpoint_dir=out/'states'
    request=dict(poses=group['poses'],warm_start=True,independent_cold_start=False,
        source_method='old whole; never incremental',old_layout_sha256=I.sha256(source),
        old_report_sha256=I.sha256(old/'search_report.json'),original_constraints_unchanged=True,
        prior_failed_cold_search_not_relabelled=True,actual_candidate_budget=3,
        code=I.hashes([Path(__file__)]),trials=[])
    for index,margin in enumerate([0.,.03125,.125]):
        trial=layout.copy()
        for k in trial.active:
            trial.directions[k]=legal_direction(layout.directions[k]+np.tan(np.radians(margin))*model.floor_normal(layout,k),model.floor_normal(layout,k))
        record=dict(index=index,world_up_margin_degrees=margin,passed=False)
        result=model.evaluate(trial);model.commit(result);record['sampled_failed_loads']=failed_count(result)
        if failed_count(result)==0:
            np.savez_compressed(out/f'layout_{index:02d}.npz',placements=trial.placements,
                directions=trial.directions,hosts=trial.hosts,active=np.array(trial.active))
            try:
                actual=model.exact(trial);write_result(actual,out/f'result_{index:02d}')
                record.update(counts=actual['counts'],volume_cm3=actual['volume_cm3'],passed=failed_count(actual)==0)
            except (RuntimeError,ValueError,AssertionError) as error:record['error']=str(error)
        request['trials'].append(record);save(out/'report.json',request)
        print('WARM_PROBE',group['id'],record,flush=True)
        if record['passed']:break
    request['complete']=True;request['passed']=any(r['passed'] for r in request['trials']);save(out/'report.json',request)


if __name__=='__main__':main()
