"""Step4.2 orchestration: continuous updates, discrete jumps, real acceptance."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import argparse
import contextlib
import os
import shutil
import time
import traceback
from co_common import *
from direction_choice import direction_choice
from translation.boundary_update import translation
from juxtapose.reuse_search import juxtapose
from continuous_support.complete_objective import CompleteObjective as CoverageObjective
from continuous_support.boundary_descent import block_direction_choice
from whole_search.fast_search import FastModel
from whole_search.model import Model,Layout
from whole_search.search import failed_count
from whole_search.common import legal_direction
from whole_pipeline import load_layout
from compact_volume import load_incumbent
from run_all import saved_groups,output_root
from whole_step4_render import render_search


def layout_data(layout):
    return dict(placements=layout.placements.tolist(),directions=layout.directions.tolist(),
                hosts=layout.hosts.tolist(),active=list(layout.active))


def restore_layout(data):
    return Layout(np.asarray(data['placements']),np.asarray(data['directions']),
                  np.asarray(data['hosts'],int),tuple(data['active']))


def source_files():
    result=[Path(__file__),Path(__file__).with_name('run.py')]
    for package in ['continuous_support','direction_choice','translation','juxtapose']:
        result+=list((HERE/'helper_func'/package).glob('*.py'))
    result += [HERE/'helper_func/whole_search/reuse_first.py', HERE/'helper_func/whole_search/search.py', HERE/'helper_func/whole_search/fast_search.py']
    return result


def preserve_sampled_feasibility(objective,before,after,record):
    if record['accepted'] and objective.covered(before):
        previous=objective.model.evaluate(before.layout)
        if failed_count(previous)==0:
            checked=objective.model.evaluate(after.layout)
            if failed_count(checked):
                record=dict(record,accepted=False,sampled_original_load_guard_rejected=True,
                    reason='Retain the all-original-load sampled feasible incumbent',
                    rejected_sampled_counts=checked['counts'])
                return before,record
    return after,record


def optimize(model,layout,out,options):
    objective=CoverageObjective(model,layout,options['quadrature_level'],options['contact_depth'])
    current=objective.evaluate(layout);history=[(current,'step4.1',{})];trace=[];jumps=0;visited_jumps=set();volume_rounds=0

    def refine(candidate,owners=None):
        records=[]
        for _ in range(options['jump_refine']):
            before=candidate
            for operation in [translation,direction_choice]:
                if operation is direction_choice:
                    candidate,record=block_direction_choice(objective,candidate,None,backtracks=options['backtracks'])
                else:candidate,record=operation(objective,candidate,backtracks=options['backtracks'])
                if record['accepted']:record['layout']=layout_data(candidate.layout)
                records.append(record)
            if candidate is before:break
        return candidate,records

    for iteration in range(options['iterations']):
        before=current;events=[];volume_phase=objective.covered(before)
        previous=current
        current,record=block_direction_choice(objective,current,None,backtracks=options['backtracks'])
        current,record=preserve_sampled_feasibility(objective,previous,current,record)
        events.append(record)
        if record['accepted']:history.append((current,'direction',record))
        previous=current
        current,record=translation(objective,current,backtracks=options['backtracks'])
        current,record=preserve_sampled_feasibility(objective,previous,current,record)
        events.append(record)
        if record['accepted']:history.append((current,'translation',record))
        relative_progress=(objective.coverage_loss(before)-objective.coverage_loss(current))/max(objective.coverage_loss(before),1e-12)
        stalled=current is before or (not objective.covered(current) and relative_progress<options.get('minimum_progress',.005))
        local_window_exhausted=(not objective.covered(current) and
            iteration+1>=options.get('initial_direction_rounds',3))
        if (stalled or local_window_exhausted) and jumps<options['jumps']:
            current,record=juxtapose(objective,current,budget=options.get('jump_screen_budget',96),
                full_budget=options['jump_trials'],refine=refine,
                excluded=visited_jumps,max_refined=options.get('jump_branches',3))
            record['trigger']=('bounded continuous repair has not covered the whole set'
                if local_window_exhausted else 'continuous stagnation or insufficient all-pose progress')
            jumps+=1;events.append(record)
            if record['accepted']:
                raw=objective.evaluate(restore_layout(record['jump_layout']))
                history.append((raw,'juxtapose',record['selected']))
                for row in record['continuous_refinement']:
                    if row.get('accepted'):
                        state=objective.evaluate(restore_layout(row['layout']))
                        history.append((state,row['operation'],row))
                if history[-1][0].layout.key()!=current.layout.key():history.append((current,'jump_refined',{}))
        row=dict(iteration=iteration+1,coverage=current.coverage.tolist(),
                 residual_loss=current.residual_loss.tolist(),maximum_residual=current.maximum_residual.tolist(),
                 estimated_material_cm3=current.volume_cm3,events=events,
                 continuous_objective_evaluations=objective.evaluations,interval_lp_calls=objective.lp_calls,
                 continuous_relative_progress=relative_progress,timing=objective.timing())
        states=out/'search_states';states.mkdir(exist_ok=True)
        np.savez_compressed(states/f'round_{iteration+1:03d}.npz',placements=current.layout.placements,
            directions=current.layout.directions,hosts=current.layout.hosts,active=np.asarray(current.layout.active))
        trace.append(row);save(out/'trace.json',trace)
        print('CONTINUOUS',iteration+1,'coverage',current.coverage.tolist(),
              'material',round(current.volume_cm3,3),'evaluations',objective.evaluations,flush=True)
        if volume_phase:volume_rounds+=1
        if objective.covered(current) and volume_rounds>=options.get('volume_rounds',2):break
        if current is before and jumps>=options['jumps']:break
    save(out/'search_report.json',dict(complete=True,all_poses_active=True,
        objective='integrated squared distance of original coupled demands to the reaction cone, then material volume',
        continuous_domain_certified=False,random_design_sampling=False,hardest_demand_search=False,
        initial_coverage=history[0][0].coverage.tolist(),final_coverage=current.coverage.tolist(),
        initial_residual_loss=history[0][0].residual_loss.tolist(),final_residual_loss=current.residual_loss.tolist(),
        final_guidance_details=current.details,iterations=len(trace),discrete_jump_calls=jumps,
        objective_evaluations=objective.evaluations,magnitude_interval_lps=objective.lp_calls,
        objective_seconds=objective.seconds,timing=objective.timing(),options=options))
    return objective,history


def save_process(model,history,selected,out):
    rows=[];directory=out/'process_states';directory.mkdir(parents=True,exist_ok=True)
    for index,(result,operation,decision) in enumerate(history[:selected+1]):
        layout=result.layout;path=directory/f'{index:03d}.npz'
        np.savez_compressed(path,placements=layout.placements,directions=layout.directions,
                            hosts=layout.hosts,active=np.asarray(layout.active),native_world=model.native)
        rows.append(dict(index=index,phase=operation,layout=str(path.relative_to(out)),
                         continuous_coverage=result.coverage.tolist(),estimated_volume_cm3=result.volume_cm3,
                         residual_loss=result.residual_loss.tolist(),maximum_residual=result.maximum_residual.tolist(),
                         geometry_verified=False,decision=decision))
    save(out/'process.json',rows)


def run_case(name,group,options):
    generation=output_root(name)/'data/whole_step4_active_run.json'
    if generation.exists():
        token=json.loads(generation.read_text())['run_token']
        previous=os.environ.get('COOPT_WHOLE_RUN_TOKEN')
        if previous is not None and previous!=token:
            return dict(id=group['id'],passed=False,cancelled=True,error='Stale cancelled batch generation')
    began=time.monotonic();base=output_root(name)/group['id'];source=base/'step4/step4.1'
    output_name=options.get('output_name','continuous')
    if not output_name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in output_name):
        raise ValueError('Output name must be a single directory name')
    out=base/'step4/step4.2'/output_name;out.mkdir(parents=True,exist_ok=True);(out/'data').mkdir(exist_ok=True)
    if options['resume'] and (out/'data/report.json').exists():
        report=I.check_report(out/'data/report.json')
        return dict(id=group['id'],passed=report['force_exit_work_passed'],reused_completed_result=True)
    if any(p.name!='data' for p in out.iterdir()) or any((out/'data').iterdir()):
        archive=base/'step4/_history'/(output_name+'_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns()%1000000000))
        shutil.copytree(out,archive)
    with (out/'data/run.log').open('w',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            report41=I.check_report(source/'data/report.json')
            assert report41['poses']==group['poses']
            generation=output_root(name)/'data/whole_step4_active_run.json'
            if generation.exists():
                os.environ['COOPT_WHOLE_RUN_TOKEN']=json.loads(generation.read_text())['run_token']
            model=FastModel(group['poses'],name,initialization_report=base/'step3/step3.1/data/report.json')
            layout=load_layout(source/'layout.npz')
            assert layout.active==tuple(range(len(group['poses'])))
            model.checkpoint_dir=out/'data/exact_states'
            model.worker_program=HERE/'helper_func/whole_search/exact_worker.py'
            baseline=None
            if report41['force_exit_work_passed']:baseline,_=load_incumbent(model,source)
            frozen=out/'data/executed_sources'/time.strftime('%Y%m%d_%H%M%S');hashes=I.hashes(source_files());aliases={}
            for relative,digest in hashes.items():
                target=frozen/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,target)
                aliases[relative]=str(target.relative_to(ROOT))
            save(out/'data/source_manifest.json',dict(code=hashes,snapshots=aliases,
                step41_report_sha256=I.sha256(source/'data/report.json'),step41_modified=False))
            objective,history=optimize(model,layout,out,options)
            # The optional finer quadrature is checked on the same layout,
            # separately from original sampled-load final acceptance.
            fine=CoverageObjective(model,layout,min(options['quadrature_level']+1,3),options['contact_depth'])
            refined=fine.evaluate(history[-1][0].layout)
            if objective.covered(history[-1][0]) and not fine.covered(refined):
                finer=refined
                for refinement in range(options.get('fine_iterations',2)):
                    previous=finer
                    finer,decision=block_direction_choice(fine,finer,None,backtracks=options['backtracks'])
                    if decision['accepted']:history.append((finer,'fine_direction',decision))
                    finer,decision=translation(fine,finer,backtracks=options['backtracks'])
                    if decision['accepted']:history.append((finer,'fine_translation',decision))
                    if fine.covered(finer) or finer is previous:break
                refined=finer
                save(out/'fine_refinement.json',dict(complete=True,iterations=refinement+1,
                    coverage=finer.coverage.tolist(),residual_loss=finer.residual_loss.tolist(),
                    maximum_residual=finer.maximum_residual.tolist(),timing=fine.timing(),extra_budget=True))
            save(out/'search_quadrature_check.json',dict(coverage=refined.coverage.tolist(),
                coarse_coverage=history[-1][0].coverage.tolist(),details=refined.details,
                continuous_domain_certified=False))
            best=baseline;selected=0;attempts=[]
            candidates=sorted(range(1,len(history)),key=lambda i:
                (not objective.covered(history[i][0]),objective.coverage_loss(history[i][0]),history[i][0].volume_cm3))
            queue=[(index,1.,history[index][0].layout) for index in candidates]
            for attempt in range(options['real_budget']):
                if not queue:break
                index,fraction,candidate=queue.pop(0)
                trial=dict(process_index=index,accepted=False,continuous_step_fraction=fraction)
                try:
                    coarse=objective.evaluate(candidate)
                    if not objective.covered(coarse):
                        trial.update(status='coarse_guidance_not_whole_feasible',coarse_coverage=coarse.coverage.tolist(),
                            continuous_quadrature_gate_passed=False,actual_geometry_evaluated=False)
                        attempts.append(trial);save(out/'actual_validation.json',attempts)
                        continue
                    integral=fine.evaluate(candidate)
                    if not fine.covered(integral):
                        trial.update(status='guidance_not_whole_feasible',fine_continuous_coverage=integral.coverage.tolist(),
                            continuous_quadrature_gate_passed=False,actual_geometry_evaluated=False)
                        attempts.append(trial);save(out/'actual_validation.json',attempts)
                        continue
                    actual=model.exact(candidate)
                    passed=failed_count(actual)==0 and all(r['passed'] for r in actual.get('actual_work_surface_checks',[]))
                    trial.update(force_exit_work_passed=passed,volume_cm3=actual['volume_cm3'])
                    integral=fine.evaluate(actual['layout']) if passed else None
                    if integral is not None:
                        trial.update(fine_continuous_coverage=integral.coverage.tolist(),
                                     continuous_quadrature_gate_passed=fine.covered(integral))
                    if passed and fine.covered(integral) and (best is None or actual['volume_cm3']<best['volume_cm3']-1e-4):
                        best=actual;selected=index
                except (RuntimeError,ValueError,AssertionError) as error:trial['error']=str(error)
                attempts.append(trial);save(out/'actual_validation.json',attempts)
                if selected==index and best is not None:break
                # Backtrack the same continuous move after a failed REAL
                # check, rather than introducing random recovery directions.
                if baseline is not None and np.array_equal(candidate.hosts,layout.hosts):
                    shorter=layout.copy()
                    shorter.placements[:,:3,3]=(candidate.placements[:,:3,3]+layout.placements[:,:3,3])/2
                    for k in shorter.active:
                        shorter.directions[k]=legal_direction(candidate.directions[k]+layout.directions[k],model.floor_normal(shorter,k))
                    state=objective.evaluate(shorter)
                    history.append((state,'actual_line_search',dict(step_fraction=fraction/2,
                        reason='backtrack continuous move after actual acceptance failure')))
                    queue.insert(0,(len(history)-1,fraction/2,shorter))
            if best is None:
                save_process(model,history,len(history)-1,out)
                save(out/'unresolved_search.json',dict(complete=True,passed=False,
                    best_guidance_layout=layout_data(history[-1][0].layout),
                    best_guidance_coverage=history[-1][0].coverage.tolist(),
                    actual_validation_attempts=attempts,continuous_domain_certified=False))
                raise RuntimeError('Continuous guidance did not obtain an actual all-load feasible result')
            for trial in attempts:trial['accepted']=trial['process_index']==selected
            save(out/'actual_validation.json',attempts);save_process(model,history,selected,out)
            final_integral=fine.evaluate(best['layout']);coarse=objective.evaluate(best['layout'])
            save(out/'quadrature_check.json',dict(coverage=final_integral.coverage.tolist(),
                coarse_coverage=coarse.coverage.tolist(),details=final_integral.details,
                evaluated_layout_is_published_actual=True,continuous_domain_certified=False))
            result=Model.save(model,best,out,dict(schema='whole_step4_v1',strategy='continuous_coverage',
                passed=True,status='pass',initialized_all_poses_together=True,
                continuous_domain_certified=False,random_design_sampling=False,hardest_demand_search=False,
                continuous_quadrature_gate_passed=fine.covered(final_integral),
                step41_baseline_retained=selected==0,selected_process_index=selected,
                final_validation_attempts=attempts,options=options,
                baseline_volume_cm3=None if baseline is None else baseline['volume_cm3'],
                all_pose_juxtapose_disabled=True,separated_fallback_used=False,
                fixture_placement_count_is_not_cost=True,seconds=time.monotonic()-began))
            result['provenance']['inputs'].update(I.hashes([source/'data/report.json',source/'layout.npz']))
            result['provenance']['code'].update({aliases[p]:v for p,v in hashes.items()})
            save(out/'report.json',result)
            data=result.copy();data['artifacts']={'../'+k:v for k,v in result['artifacts'].items()}
            save(out/'data/report.json',data)
            render_search(model,out,result)
            rendered=json.loads((out/'data/render.json').read_text())
            for row in rendered['process_states']:
                row['process_force_evaluation']='integrated reaction-cone distance and moving surface contact polygons'
            rendered.update(continuous_domain_certified=False,
                process_evaluation_is_continuous_quadrature=True)
            rendered['provenance']['code'].update({aliases[str(Path(__file__).relative_to(ROOT))]:I.sha256(Path(__file__))})
            save(out/'data/render.json',rendered)
            for filename in ['process.png','final_result.png']:
                if filename in result['artifacts']:result['artifacts'][filename]=I.sha256(out/filename)
            save(out/'report.json',result)
            data=result.copy();data['artifacts']={'../'+k:v for k,v in result['artifacts'].items()}
            save(out/'data/report.json',data)
            I.check_report(out/'data/report.json')
            return dict(id=group['id'],passed=True,volume_cm3=result['volume_cm3'],
                        baseline_retained=selected==0,seconds=time.monotonic()-began)
        except Exception as error:
            traceback.print_exc();save(out/'data/failure.json',dict(complete=True,passed=False,error=str(error)))
            return dict(id=group['id'],passed=False,error=str(error),seconds=time.monotonic()-began)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('object',nargs='?',default='B')
    parser.add_argument('--sets',nargs='+',required=True);parser.add_argument('--iterations',type=int,default=8)
    parser.add_argument('--jumps',type=int,default=8);parser.add_argument('--jump-trials',type=int,default=6)
    parser.add_argument('--jump-refine',type=int,default=3)
    parser.add_argument('--jump-screen-budget',type=int,default=96);parser.add_argument('--backtracks',type=int,default=4)
    parser.add_argument('--jump-branches',type=int,default=3)
    parser.add_argument('--initial-direction-rounds',type=int,default=3)
    parser.add_argument('--volume-rounds',type=int,default=2)
    parser.add_argument('--quadrature-level',type=int,choices=[1,2],default=1)
    parser.add_argument('--contact-depth',type=int,choices=[1,2],default=1)
    parser.add_argument('--real-budget',type=int,default=3);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--output-name',default='reuse_gradient_v1')
    args=parser.parse_args();groups=saved_groups(args.object)
    if args.sets:groups=[g for g in groups if g['id'] in args.sets]
    if not groups:parser.error('No existing initialized set matches')
    batch_path=output_root(args.object)/'data'/(args.output_name+'_batch.json')
    if batch_path.exists():parser.error('Preserve this experiment; use a distinct --output-name for another run')
    token=os.urandom(16).hex();os.environ['COOPT_WHOLE_RUN_TOKEN']=token
    generation=output_root(args.object)/'data/whole_step4_active_run.json'
    save(generation,dict(object=args.object,run_token=token,status='running_reuse_gradient',
        experiment=args.output_name,old_52_case_batch_remains_stopped=True))
    rows=[]
    for group in groups:
        row=run_case(args.object,group,vars(args));rows.append(row)
        save(batch_path,dict(complete=False,results=rows,options=vars(args)))
        print('CONTINUOUS CASE',row,flush=True)
    save(batch_path,dict(complete=True,results=rows,options=vars(args)))
    save(generation,dict(object=args.object,run_token=token,status='reuse_gradient_complete',
        experiment=args.output_name,old_52_case_batch_remains_stopped=True))
    if not all(r['passed'] for r in rows):raise SystemExit(2)


if __name__=='__main__':main()
