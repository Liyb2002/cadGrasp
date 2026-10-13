"""Cold Step4 search with persistent layouts and force/base feasibility."""
from pathlib import Path
import sys
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
COOPT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(COOPT/'helper_func'))
import _bootstrap
import argparse
import json
import time
import traceback
import shutil
import multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
from oracle_search import OracleModel,OracleSearch
from whole_pipeline import load_layout
from co_common import ROOT,I,S,save,trimesh
from whole_step4_render import render_search
from render_pose_oracle import export_pose_constraints
import artifact_store as A


def read_upstream(path):
    """Frozen upstream artifacts/physical inputs, not current search source."""
    report=json.loads(path.read_text());assert report['complete']
    I.check_hashes(report['provenance']['inputs'])
    for filename,digest in report.get('artifacts',{}).items():
        assert I.sha256(path.parent/filename)==digest,filename
    return report


def worker(job):
    label,options=job;base=A.result_root('B')/label
    out=base/'step4/step4.2'/options['output_name'];out.mkdir(parents=True,exist_ok=False);(out/'data').mkdir()
    began=time.monotonic();result=dict(id=label,passed=False,force_passed=False,base_oracle_passed=False)
    try:
        source=base/'step4/step4.1';original=read_upstream(source/'data/report.json')
        initialized=base/'step3/step3.1';read_upstream(initialized/'data/report.json')
        model=OracleModel(original['poses'],'B',demand_nodes=32,
                         oracle_work_mode=options.get('oracle_work_mode','normal'))
        seed=initialized/'wrapped_support.obj'
        model.registered_seed=S.solid(trimesh.load(seed,force='mesh',process=False))
        model.inputs += [seed,initialized/'data/report.json',source/'layout.npz']
        search=OracleSearch(model,out,iterations=options['iterations'],finalists=3,branch_rounds=2,seed=42,
                           frontier_width=options.get('frontier_width',4),
                           search_policy=options.get('search_policy','frontier'))
        search.initialization_budget=options.get('initialization_budget',96)
        search.allow_oracle_fallback=options.get('allow_oracle_fallback',False)
        search.allow_coupled_adjustment=options.get('allow_coupled_adjustment',False)
        search.allow_group_seating=options.get('allow_group_seating',False)
        search.allow_joint_reseating=options.get('allow_joint_reseating',False)
        search.allow_placement_guidance=options.get('allow_placement_guidance',False)
        search.allow_height_repair=options.get('allow_height_repair',False)
        search.stop_when_feasible=options.get('stop_when_feasible',False)
        search.allow_native_sliding=options.get('allow_native_sliding',False)
        search.warm_checkpoint_experiment=options.get('warm_checkpoint_experiment')
        search.initialization_seeds=options.get('initialization_seeds')
        search.material_warm_layouts=[]
        for experiment in options.get('warm_start_experiments',[]):
            path=base/'step4/step4.2'/experiment/'layout.npz'
            if not path.exists():
                path=base/'step4/step4.2'/experiment/'interrupted_layout.npz'
            if path.exists():
                search.material_warm_layouts.append((str(path.relative_to(ROOT)),load_layout(path)))
                model.inputs.append(path)
        search.screen_budget=96;search.capture_process=True;search.volume_rounds=2
        layout=load_layout(source/'layout.npz')
        current=search.initialize_exact(layout,'saved_step4.1_registered_initial');search.baseline(current)
        initial=current['counts']
        if options.get('resume'):
            current=model.evaluate(load_layout(base/'step4/step4.2'/options['resume']/'layout.npz'))
            model.commit(current)
        current=search.solve_frontier(current)
        if (search.search_policy not in ('material','population32','population_soft') and current['base_oracle'].get('connected_passed',False)
                and current['base_oracle']['passed']):
            current=search.restore_reuse(current);current=search.polish_volume(current,1)
        mode=('failure_weighted_soft_count32' if search.search_policy=='population_soft' else
              'fixed_population32' if search.search_policy=='population32' else
              'shared_material_mixed_search' if search.search_policy=='material' else
              'multi_layout_force_base_frontier' if search.frontier_width>1 else 'whole_gradient_base_oracle')
        report=search.save(current,mode,began,dict(source='saved Step4.1',
            old_answers_used_as_start=bool(search.material_warm_layouts),
            warm_start_layouts=[p for p,q in search.material_warm_layouts],
            resume_own_experiment=options.get('resume')),initial_counts=initial)
        data=dict(report);data['artifacts']={'../'+k:v for k,v in report['artifacts'].items()}
        save(out/'data/report.json',data)
        if report['mesh_exported']:
            try:render_search(model,out,report)
            except Exception as error:report['render_error']=str(error)
        result.update(passed=report['passed'],force_passed=report['force_passed'],
            base_oracle_passed=report['base_oracle_passed'],pose_count=len(original['poses']),
            volume_cm3=report['volume_cm3'],counts=report['counts'],
            oracle_status=report['base_oracle']['status'],oracle_seconds=model.base_oracle.seconds,
            oracle_queries=model.base_oracle.queries,component_queries=model.base_oracle.component_queries,
            maximum_object_height_m=max(model.workpiece_height(current['layout'],k) for k in current['layout'].active),
            state_count=len(set(current['layout'].hosts.tolist())),experiment=options['output_name'])
        result.update(frontier_width=search.frontier_width,frontier_search=report['frontier_search'],
                      oracle_work_access_mode=model.base_oracle.work_mode)
        if search.search_policy in ('material','population32','population_soft'):
            result.update(mixed_target_satisfied=report['mixed_target_satisfied'],
                mixed_target_applicable=report['mixed_target_applicable'],
                rotating_reuse_pose_count=report['rotating_reuse_pose_count'],
                juxtaposed_pose_count=report['juxtaposed_pose_count'],
                solid_constructed_during_search=report['solid_constructed_during_search'])
        save(out/'report.json',report)
        if report['base_oracle_passed']:
            # Keep the final task-owned cone equations and display excerpts.
            # Videos are generated on request; no search/force replay here.
            oracle_out=A.VISUALS/'B'/label/'step4/step4.2'/options['output_name']/'oracle'
            export_pose_constraints(model,current['layout'],oracle_out,model.base_oracle)
    except Exception as error:
        traceback.print_exc();result.update(status='unresolved',error=str(error))
    A.publish_visuals(out)
    A.cleanup_stage(out)
    result['seconds']=time.monotonic()-began;save(out/'run_result.json',result)
    print('ORACLE RESULT',json.dumps(result),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sets',nargs='+',default=['all']);parser.add_argument('--jobs',type=int,default=3)
    parser.add_argument('--iterations',type=int,default=6);parser.add_argument('--resume')
    parser.add_argument('--search-policy',choices=['frontier','material','population32','population_soft'],default='frontier')
    parser.add_argument('--allow-oracle-fallback',action='store_true')
    parser.add_argument('--allow-coupled-adjustment',action='store_true')
    parser.add_argument('--allow-group-seating',action='store_true')
    parser.add_argument('--allow-joint-reseating',action='store_true')
    parser.add_argument('--allow-placement-guidance',action='store_true')
    parser.add_argument('--allow-height-repair',action='store_true')
    parser.add_argument('--stop-when-feasible',action='store_true')
    parser.add_argument('--allow-native-sliding',action='store_true')
    parser.add_argument('--initialization-budget',type=int,default=96)
    parser.add_argument('--warm-start-experiments',nargs='*',default=[])
    parser.add_argument('--warm-checkpoint-experiment')
    parser.add_argument('--initialization-seeds',help='Explicit per-case layout seed manifest; inputs are frozen and recorded')
    parser.add_argument('--min-poses',type=int,default=1)
    parser.add_argument('--max-poses',type=int,default=1000000)
    parser.add_argument('--frontier-width',type=int,default=4,
                        help='Number of persistent complete layouts; 1 selects the prior single-layout search')
    parser.add_argument('--oracle-work-mode',choices=['normal','angular'],default='normal')
    parser.add_argument('--output-name',default='base_oracle_v1');args=parser.parse_args()
    if args.frontier_width<1:
        parser.error('--frontier-width must be positive')
    if args.search_policy in ('population32','population_soft'):
        if args.frontier_width!=32:
            parser.error('population32 requires --frontier-width 32')
        if args.resume or args.initialization_seeds or args.warm_start_experiments or args.warm_checkpoint_experiment:
            parser.error('population32 starts from registered inputs; warm/resume layouts are not allowed')
    base=A.result_root('B')
    sets=args.sets
    if sets==['all']:
        sets=[str(p.relative_to(base)).split('/step4/')[0] for p in base.glob('*/step4/step4.1/data/report.json')]
        sets += [str(p.relative_to(base)).split('/step4/')[0] for p in (base/'illegal').glob('*/step4/step4.1/data/report.json')]
        sets.sort(key=lambda label:(-len(json.loads((base/label/'step4/step4.1/data/report.json').read_text())['poses']),label))
    sets=[label for label in sets if args.min_poses<=len(json.loads(
        (base/label/'step4/step4.1/data/report.json').read_text())['poses'])<=args.max_poses]
    if not sets:
        parser.error('No saved pose sets in the requested range')
    directory=base/'data'/args.output_name;directory.mkdir(parents=True,exist_ok=False)
    sources=list((COOPT/'helper_func').rglob('*.py'))+list((COOPT/'vis_func').glob('*.py'))+list((COOPT/'step4.2').glob('*.py'))
    sources.append(COOPT/'step5.3/deterministic_base.py')
    hashes=I.hashes(sources)
    for relative in hashes:
        target=directory/'executed_sources'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,target)
    save(directory/'sources.json',hashes)
    protected={}
    for label in sets:
        for stage in ['step3','step4/step4.1']:
            protected.update(I.hashes([p for p in (base/label/stage).rglob('*') if p.is_file()]))
    save(directory/'protected.json',protected)
    batch=dict(complete=False,sets=sets,group_count=len(sets),options=vars(args),results=[],
        old_incumbents_accepted_without_oracle=False,intermediate_rotations_constrained=False)
    save(directory/'batch.json',batch);start=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(worker,(label,vars(args))) for label in sets]
        for future in as_completed(futures):
            batch['results'].append(future.result());batch['seconds']=time.monotonic()-start
            batch['passed']=sum(r['passed'] for r in batch['results']);save(directory/'batch.json',batch)
    I.check_hashes(protected);I.check_hashes(hashes)
    batch.update(complete=True,protected_unchanged=True,executed_sources_unchanged=True,seconds=time.monotonic()-start)
    save(directory/'batch.json',batch);A.finish_batch(directory)
    for label in sets:
        (base/label/'step4/step4.2'/args.output_name/'run_result.json').unlink(missing_ok=True)
    print('ALL ORACLE',batch['passed'],'/',len(sets),flush=True)


if __name__=='__main__':main()
