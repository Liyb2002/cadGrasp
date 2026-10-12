"""Strictly replay saved sampled loads; continue unresolved local searches."""
import argparse
import hashlib
import json
import shutil
import time
from common import *
from case_sets import CASES,case_directory
from fast_search import FastModel,FastReuseSearch
from mesh_media import load_layout
from reuse_first import registered
from search import failed_count


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--case',choices=list(CASES),required=True)
    parser.add_argument('--method',choices=['whole','incremental'],required=True)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--polish',action='store_true',help='Preserve feasibility while restoring reuse and reducing volume')
    args=parser.parse_args()
    out=case_directory(args.root.resolve(),args.case)/args.method
    path=out/'search_report.json';old=json.loads(path.read_text())
    began=time.monotonic();model=FastModel(old['requested_poses'])
    current=model.evaluate(load_layout(out/'sampled_layout.npz'));model.commit(current)
    check=dict(complete=True,sampled_force_passed=failed_count(current)==0,
        full_requested_set_evaluated=len(current['layout'].active)==len(model.poses),
        counts={model.poses[k]:int(mask.sum()) for k,mask in current['masks'].items()},
        original_report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        sampled_layout_sha256=hashlib.sha256((out/'sampled_layout.npz').read_bytes()).hexdigest(),
        classifier_sha256=hashlib.sha256((HERE/'code/classify.py').read_bytes()).hexdigest(),
        classifiers={model.poses[k]:info for k,info in current['classifiers'].items()},
        tolerance=2e-9,original_equations=True,original_work_regions=True,
        exact_pressure_acceptance=False,seconds=time.monotonic()-began)
    C.save(out/('pre_polish_sampled_recheck.json' if args.polish else 'strict_sampled_recheck.json'),check)
    print('STRICT SAMPLED CHECK',args.case,args.method,failed_count(current),flush=True)
    if args.polish:assert check['sampled_force_passed'] and check['full_requested_set_evaluated']
    if args.check_only or (not args.polish and failed_count(current)==0 and check['full_requested_set_evaluated']):return
    history=out/'_history'/('before_feasible_polish' if args.polish else 'before_local_continuation')
    if history.exists():
        history=out/'_history'/('before_local_continuation_'+str(time.time_ns()))
    history.mkdir(parents=True,exist_ok=False)
    for name in ['search_report.json','sampled_layout.npz','process.json','trace.json','insertion_stages.json']:
        if (out/name).exists():shutil.copy2(out/name,history/name)
    for name in ['classify.py','fast_search.py','reuse_first.py','repair_fast_search.py']:
        shutil.copy2(HERE/'code'/name,history/name)
    search=FastReuseSearch(model,out,iterations=8,finalists=3,branch_rounds=3)
    search.search_only=True;search.capture_process=True;search.screen_budget=96;search.volume_rounds=2
    search.events=old['events'];model.sample_calls=old['sample_evaluations']+1
    search.process_rows=json.loads((out/'process.json').read_text())
    search.process_layouts=[load_layout(out/row['layout']) for row in search.process_rows]
    search.process_started=began-old['search_seconds']
    movable=[k for k in current['layout'].active if not registered(current['layout'],k)]
    if args.polish:
        current=search.restore_reuse(current)
        current=search.polish_volume(current,2)
    else:
        current=search.refine(current,3,anchor=None,translation_guests=movable,
                              phase='work_cone_final_local_repair')
        if failed_count(current):
            current=search.solve_frontier(current)
    mode='joint' if args.method=='whole' else 'incremental'
    extra={key:old[key] for key in ['insertion_stages','every_sampled_insertion_passed'] if key in old}
    if mode=='incremental':
        stages=extra['insertion_stages']
        stages[-1]=dict(stages[-1],passed=failed_count(current)==0,
            counts=current['counts'],retried=True,lost_existing_passed_loads=0 if failed_count(current)==0 else None)
        extra['every_insertion_passed']=all(row['passed'] for row in stages)
        C.save(out/'insertion_stages.json',stages)
        search.process_stage(current,stages[-1])
    new=search.save(current,mode,began-old['search_seconds'],{},**extra)
    if mode=='incremental' and failed_count(current)==0 and len(current['layout'].active)<len(model.poses):
        new=search.incremental(start_prefix=out/'sampled_layout.npz')
        new['search_seconds']=old['search_seconds']+(time.monotonic()-began)
    for key in ['strategy','requested_poses','requested_pose_count','independent_initialization','inputs','search_source_manifest']:
        if key in old:new[key]=old[key]
    new.update(original_search_seconds=old['search_seconds'],
        additional_local_continuation_wall_seconds=time.monotonic()-began,
        continuation_includes_preparation=True,
        continuation_sources={name:hashlib.sha256((history/name).read_bytes()).hexdigest()
                              for name in ['classify.py','fast_search.py','reuse_first.py','repair_fast_search.py']},
        continuation_kind='feasible_volume_polish' if args.polish else 'load_recovery',
        continuation_branch_refinement_rounds=0 if args.polish else 3,
        previous_attempt=str(history.relative_to(out)),
        sampled_recheck_before_continuation=check,
        final_continuation_failed_load_count=sum(32768-v for v in new['counts'].values()))
    C.save(path,new)
    if args.polish:
        check.update(sampled_force_passed=failed_count(current)==0,
            counts={model.poses[k]:int(mask.sum()) for k,mask in current['masks'].items()},
            classifiers={model.poses[k]:info for k,info in current['classifiers'].items()},
            original_report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            sampled_layout_sha256=hashlib.sha256((out/'sampled_layout.npz').read_bytes()).hexdigest(),
            seconds=time.monotonic()-began)
        C.save(out/'strict_sampled_recheck.json',check)
    print('LOCAL CONTINUATION',args.case,args.method,new['sampled_force_passed'],new['pose_count'],
          'extra seconds',round(time.monotonic()-began,2),flush=True)


if __name__=='__main__':main()
