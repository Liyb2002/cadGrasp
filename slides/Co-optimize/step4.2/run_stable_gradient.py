"""Cold whole-gradient search; force/torque PASS without final geometry replay."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import argparse
import json
import os
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing
from co_common import HERE, ROOT, I, save
from whole_pipeline import load_layout
from whole_search.saved_initialization import check_saved_report
from continuous_support.stable_gradient import StableGradientModel as GradientModel, StableGradientSearch as GradientSearch
from whole_search.search import failed_count
from whole_step4_render import render_search


def worker(arg):
    label,options,token=arg;os.environ['COOPT_WHOLE_RUN_TOKEN']=token
    generation=HERE/'output/B/data/whole_step4_active_run.json'
    if generation.exists() and json.loads(generation.read_text())['run_token']!=token:
        return dict(id=label,passed=False,status='cancelled_before_initialization',complete=False)
    base=HERE/'output/B'/label;out=base/'step4/step4.2'/options['output_name']
    out.mkdir(parents=True,exist_ok=False);(out/'data').mkdir()
    if options.get('worker_cpus',0):
        allowed=sorted(os.sched_getaffinity(0));identity=multiprocessing.current_process()._identity[0]-1
        size=options['worker_cpus'];start=(identity*size)%len(allowed)
        os.sched_setaffinity(0,{allowed[(start+j)%len(allowed)] for j in range(size)})
    began=time.monotonic();result=dict(id=label,passed=False,old_answers_used_as_start=False,worker_cpu_affinity=sorted(os.sched_getaffinity(0)))
    try:
        source=base/'step4/step4.1';poses=check_saved_report(source/'data/report.json')['poses']
        model=GradientModel(poses,'B',initialization_report=base/'step3/step3.1/data/report.json',
                            demand_nodes=options['demand_nodes'])
        search=GradientSearch(model,out,iterations=options['iterations'],finalists=3,
                              branch_rounds=2,seed=42)
        search.screen_budget=96;search.capture_process=True;search.volume_rounds=2
        layout=load_layout(source/'layout.npz')
        current=search.initialize_exact(layout,'saved_step4.1_registered_initial')
        initial=current['counts'];search.baseline(current)
        if options.get('resume_experiment'):
            own=base/'step4/step4.2'/options['resume_experiment']
            current=model.evaluate(load_layout(own/'sampled_layout.npz'));model.commit(current)
            search.process_snapshot(current,'resume_own_gradient_state',dict(source=str(own.relative_to(ROOT)),
                additional_budget=True,old_optimized_answers_used=False))
            current=search.solve_frontier(current,anchor=None)
        else:current=search.solve_frontier(current,anchor=None)
        current=search.restore_reuse(current)
        current=search.polish_volume(current,2)
        report=search.save(current,'whole_gradient',began,dict(source='saved Step4.1',old_answers_used_as_start=False),
                           initial_counts=initial,passed=failed_count(current)==0,
                           resume_experiment=options.get('resume_experiment'),additional_continuation_budget=bool(options.get('resume_experiment')),
                           algorithm='whole_gradient_fixed_branch_measure_with_bounded_paired_jumps',
                           demand_nodes=options['demand_nodes'])
        # Preserve executed sources in this run's batch directory, including
        # sources changed AFTER a finished pilot in a subsequent revision.
        manifest=json.loads((HERE/'output/B/data'/options['output_name']/'source_manifest.json').read_text())
        report['provenance']['code'].update(manifest['snapshots'])
        model.timing.update(shared_geometry_seconds=model.geometry_cache.seconds,
                            shared_sweep_queries=model.geometry_cache.sweep_queries,work_query_points=model.geometry_cache.work_points)
        save(out/'report.json',report)
        data=report.copy();data['artifacts']={'../'+k:v for k,v in report['artifacts'].items()}
        save(out/'data/report.json',data)
        if report['mesh_exported']:
            try:render_search(model,out,report)
            except Exception as error:
                report['render_error']=str(error);save(out/'report.json',report)
                data=report.copy();data['artifacts']={'../'+k:v for k,v in report['artifacts'].items()}
                save(out/'data/report.json',data)
        I.check_report(out/'data/report.json')
        gradients=[t for event in search.events for t in event.get('trials',[])
                   if t.get('accepted') and 'gradient' in t.get('operation','')]
        result.update(passed=report['force_passed'],force_passed=report['force_passed'],
                      acceptance=report['acceptance'],volume_measure=report['volume_measure'],
                      mesh_exported=report['mesh_exported'],volume_cm3=report['volume_cm3'],counts=report['counts'],
                      accepted_gradient_steps=len(gradients),gradient_operations=[t['operation'] for t in gradients],
                      search_seconds=report['search_seconds'],validation_seconds=report['validation_seconds'],
                      timing=report['timing'],rotating_reuse_pose_count=report['rotating_reuse_pose_count'],
                      juxtaposed_pose_count=report['juxtaposed_pose_count'])
    except Exception as error:
        traceback.print_exc();result.update(error=str(error),status='unresolved')
    result.update(seconds=time.monotonic()-began,complete=True)
    save(out/'run_result.json',result);print('RESULT',result,flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sets',nargs='+',required=True)
    parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--worker-cpus',type=int,default=6)
    parser.add_argument('--iterations',type=int,default=10)
    parser.add_argument('--resume-experiment')
    parser.add_argument('--demand-nodes',type=int,default=32)
    parser.add_argument('--output-name',default='stable_gradient_xyz_force_v3')
    args=parser.parse_args();root=HERE/'output/B';directory=root/'data'/args.output_name
    directory.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),HERE/'vis_func/whole_step4_render.py']
    for folder in ['continuous_support','juxtapose','whole_search','translation']:
        sources+=list((HERE/'helper_func'/folder).glob('*.py'))
    sources=[s for s in sources if s.exists()];hashes=I.hashes(sources);aliases={}
    for relative,digest in hashes.items():
        target=directory/'executed_sources'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,target);aliases[str(target.relative_to(ROOT))]=digest
    save(directory/'source_manifest.json',dict(code=hashes,snapshots=aliases))
    protected={}
    for label in args.sets:
        for stage in ['step3','step4/step4.1']:
            for p in (root/label/stage).rglob('*'):
                if p.is_file():protected[str(p.relative_to(ROOT))]=I.sha256(p)
    for name in ['continuous_step4_batch.json','continuous_step42_batch.json','continuous_large_sets_v5/batch.json']:
        p=root/'data'/name
        if p.exists():protected[str(p.relative_to(ROOT))]=I.sha256(p)
    save(directory/'protected.json',protected)
    generation=root/'data/whole_step4_active_run.json'
    if generation.exists():shutil.copy2(generation,directory/'previous_generation.json')
    token=json.loads(generation.read_text())['run_token'];save(generation,dict(run_token=token,status='running_fast_gradient',
        experiment=args.output_name,old_52_case_batch_remains_stopped=True,v5_batch_remains_stopped=True))
    batch=dict(complete=False,groups=args.sets,options=vars(args),results=[],old_answers_used_as_start=False)
    began=time.monotonic();save(directory/'batch.json',batch)
    with ProcessPoolExecutor(max_workers=args.jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(worker,(label,vars(args),token)) for label in args.sets]
        for future in as_completed(futures):
            batch['results'].append(future.result());batch['seconds']=time.monotonic()-began
            save(directory/'batch.json',batch)
    for relative,digest in protected.items():assert I.sha256(ROOT/relative)==digest,relative
    for relative,digest in hashes.items():assert I.sha256(ROOT/relative)==digest,relative
    batch.update(complete=True,protected_unchanged=True,executed_sources_unchanged=True,seconds=time.monotonic()-began)
    save(directory/'batch.json',batch)
    save(generation,dict(run_token=token,status='fast_gradient_complete',experiment=args.output_name,
        old_52_case_batch_remains_stopped=True,v5_batch_remains_stopped=True))
    print('COMPLETE',sum(r['passed'] for r in batch['results']),'/',len(args.sets),flush=True)


if __name__=='__main__':main()
