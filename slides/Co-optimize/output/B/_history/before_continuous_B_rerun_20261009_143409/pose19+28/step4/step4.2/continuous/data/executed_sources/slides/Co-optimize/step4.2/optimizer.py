"""Step4.2 orchestration: continuous updates, discrete jumps, real acceptance."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import argparse
import contextlib
import shutil
import time
import traceback
from co_common import *
from direction_choice import direction_choice
from translation import translation
from juxtapose import juxtapose
from continuous_support.objective import CoverageObjective
from whole_search.fast_search import FastModel
from whole_search.model import Model,Layout
from whole_search.search import failed_count
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
    return result


def optimize(model,layout,out,options):
    objective=CoverageObjective(model,layout,options['quadrature_level'],options['contact_depth'])
    current=objective.evaluate(layout);history=[(current,'step4.1',{})];trace=[];jumps=0

    def refine(candidate):
        records=[]
        for _ in range(options['jump_refine']):
            for operation in [direction_choice,translation]:
                candidate,record=operation(objective,candidate,backtracks=options['backtracks'])
                if record['accepted']:record['layout']=layout_data(candidate.layout)
                records.append(record)
        return candidate,records

    for iteration in range(options['iterations']):
        before=current;events=[]
        current,record=direction_choice(objective,current,backtracks=options['backtracks'])
        events.append(record)
        if record['accepted']:history.append((current,'direction',record))
        current,record=translation(objective,current,backtracks=options['backtracks'])
        events.append(record)
        if record['accepted']:history.append((current,'translation',record))
        if current is before and jumps<options['jumps']:
            current,record=juxtapose(objective,current,budget=options['jump_trials'],refine=refine)
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
                 estimated_material_cm3=current.volume_cm3,events=events,
                 continuous_objective_evaluations=objective.evaluations,interval_lp_calls=objective.lp_calls)
        trace.append(row);save(out/'trace.json',trace)
        print('CONTINUOUS',iteration+1,'coverage',current.coverage.tolist(),
              'material',round(current.volume_cm3,3),'evaluations',objective.evaluations,flush=True)
        if current is before:break
    save(out/'search_report.json',dict(complete=True,all_poses_active=True,
        objective='integrated original continuous demand coverage, then material volume',
        continuous_domain_certified=False,random_design_sampling=False,hardest_demand_search=False,
        initial_coverage=history[0][0].coverage.tolist(),final_coverage=current.coverage.tolist(),
        final_guidance_details=current.details,iterations=len(trace),discrete_jump_calls=jumps,
        objective_evaluations=objective.evaluations,magnitude_interval_lps=objective.lp_calls,
        objective_seconds=objective.seconds,options=options))
    return objective,history


def save_process(model,history,selected,out):
    rows=[];directory=out/'process_states';directory.mkdir(parents=True,exist_ok=True)
    for index,(result,operation,decision) in enumerate(history[:selected+1]):
        layout=result.layout;path=directory/f'{index:03d}.npz'
        np.savez_compressed(path,placements=layout.placements,directions=layout.directions,
                            hosts=layout.hosts,active=np.asarray(layout.active),native_world=model.native)
        rows.append(dict(index=index,phase=operation,layout=str(path.relative_to(out)),
                         continuous_coverage=result.coverage.tolist(),estimated_volume_cm3=result.volume_cm3,
                         geometry_verified=False,decision=decision))
    save(out/'process.json',rows)


def run_case(name,group,options):
    began=time.monotonic();base=output_root(name)/group['id'];source=base/'step4/step4.1'
    out=base/'step4/step4.2/continuous';out.mkdir(parents=True,exist_ok=True);(out/'data').mkdir(exist_ok=True)
    if options['resume'] and (out/'data/report.json').exists():
        report=I.check_report(out/'data/report.json')
        return dict(id=group['id'],passed=report['force_exit_work_passed'],reused_completed_result=True)
    if (out/'search_report.json').exists():
        archive=base/'step4/_history'/('continuous_'+time.strftime('%Y%m%d_%H%M%S'))
        shutil.copytree(out,archive)
    with (out/'data/run.log').open('w',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            report41=I.check_report(source/'data/report.json')
            assert report41['poses']==group['poses']
            model=FastModel(group['poses'],name,initialization_report=base/'step3/step3.1/data/report.json')
            layout=load_layout(source/'layout.npz')
            assert layout.active==tuple(range(len(group['poses'])))
            model.checkpoint_dir=out/'data/exact_states'
            model.worker_program=HERE/'helper_func/whole_search/exact_worker.py'
            baseline=None
            if report41['force_exit_work_passed']:baseline,_=load_incumbent(model,source)
            frozen=out/'data/executed_sources';hashes=I.hashes(source_files());aliases={}
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
            save(out/'quadrature_check.json',dict(coverage=refined.coverage.tolist(),
                coarse_coverage=history[-1][0].coverage.tolist(),details=refined.details,
                continuous_domain_certified=False))
            best=baseline;selected=0;attempts=[]
            candidates=sorted(range(1,len(history)),key=lambda i:
                (not objective.covered(history[i][0]),objective.coverage_loss(history[i][0]),history[i][0].volume_cm3))
            for index in candidates[:options['real_budget']]:
                trial=dict(process_index=index,accepted=False)
                try:
                    actual=model.exact(history[index][0].layout)
                    passed=failed_count(actual)==0 and all(r['passed'] for r in actual.get('actual_work_surface_checks',[]))
                    trial.update(force_exit_work_passed=passed,volume_cm3=actual['volume_cm3'])
                    if passed and (best is None or actual['volume_cm3']<best['volume_cm3']-1e-4):
                        best=actual;selected=index
                except (RuntimeError,ValueError,AssertionError) as error:trial['error']=str(error)
                attempts.append(trial);save(out/'actual_validation.json',attempts)
            if best is None:raise RuntimeError('Continuous guidance did not obtain an actual all-load feasible result')
            for trial in attempts:trial['accepted']=trial['process_index']==selected
            save(out/'actual_validation.json',attempts);save_process(model,history,selected,out)
            result=Model.save(model,best,out,dict(schema='whole_step4_v1',strategy='continuous_coverage',
                passed=True,status='pass',initialized_all_poses_together=True,
                continuous_domain_certified=False,random_design_sampling=False,hardest_demand_search=False,
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
            I.check_report(out/'data/report.json')
            return dict(id=group['id'],passed=True,volume_cm3=result['volume_cm3'],
                        baseline_retained=selected==0,seconds=time.monotonic()-began)
        except Exception as error:
            traceback.print_exc();save(out/'data/failure.json',dict(complete=True,passed=False,error=str(error)))
            return dict(id=group['id'],passed=False,error=str(error),seconds=time.monotonic()-began)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('object',nargs='?',default='B')
    parser.add_argument('--sets',nargs='+');parser.add_argument('--iterations',type=int,default=6)
    parser.add_argument('--jumps',type=int,default=2);parser.add_argument('--jump-trials',type=int,default=4)
    parser.add_argument('--jump-refine',type=int,default=1);parser.add_argument('--backtracks',type=int,default=4)
    parser.add_argument('--quadrature-level',type=int,choices=[1,2],default=1)
    parser.add_argument('--contact-depth',type=int,choices=[1,2],default=1)
    parser.add_argument('--real-budget',type=int,default=2);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();groups=saved_groups(args.object)
    if args.sets:groups=[g for g in groups if g['id'] in args.sets]
    if not groups:parser.error('No existing initialized set matches')
    rows=[]
    for group in groups:
        row=run_case(args.object,group,vars(args));rows.append(row)
        save(output_root(args.object)/'data/continuous_step42_batch.json',dict(complete=False,results=rows,options=vars(args)))
        print('CONTINUOUS CASE',row,flush=True)
    save(output_root(args.object)/'data/continuous_step42_batch.json',dict(complete=True,results=rows,options=vars(args)))
    if not all(r['passed'] for r in rows):raise SystemExit(2)


if __name__=='__main__':main()
