"""Continue a retained 32/48-slot campaign with bounded actual-core trims.

Registered initialization and the population trace belong to the parent run.
Extra adjustments are recorded separately; this is not another cold benchmark.
"""
from pathlib import Path
import argparse,json,os,shutil,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[3];COOPT=ROOT/'slides/Co-optimize';BASE=COOPT/'data/results/B'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment');p.add_argument('--source-experiment',required=True)
    p.add_argument('--source-system',default='connected_core_all52_20261013_10')
    p.add_argument('--sets',nargs='+',required=True);p.add_argument('--maximum-adjustments',type=int,default=4)
    p.add_argument('--source-checkpoint',help='Saved process-state NPZ relative to the current campaign stage')
    args=p.parse_args();os.environ['COOPT_REPO_ROOT']=str(ROOT)
    frozen=BASE/'data'/('system_'+args.source_system)/'frozen_sources/slides/Co-optimize'
    sys.path.insert(0,str(frozen/'step4.2'))
    from run_oracle import OracleModel,OracleSearch,load_layout,read_upstream,S,trimesh,save,I,A
    from material_evaluator import MaterialEvaluator
    from whole_search.search import failed_count
    from whole_search.common import tangent_frame
    from continuous_support.fast_gradient import turn
    from translation.layout_update import shift,translation_frame
    from whole_search.reuse_first import registered
    from fixture_state import installed_state
    directory=BASE/'data'/args.experiment;directory.mkdir(exist_ok=False)
    controller=directory/'immutable_controller.py'
    shutil.copy2(Path(__file__),controller);controller.chmod(0o444)
    batch=dict(complete=False,sets=args.sets,group_count=len(args.sets),results=[],passed=0,
        population_size=None,rounds=8,warm_start=False,post_search_continuation=True,
        options=dict(search_policy='population_soft',require_connected_core=True,
            allow_oracle_fallback=True,stop_when_feasible=True))
    save(directory/'batch.json',batch)
    for label in args.sets:
        began=time.monotonic();base=BASE/label;parent=base/'step4/step4.2'/args.source_experiment
        initial_parent=base/'step4/step4.1'/args.source_experiment
        population_size=json.loads((initial_parent/'population.json').read_text())['population_size']
        if population_size not in [32,48]:raise ValueError('Expected current campaign 32/48 slots')
        if batch['population_size'] not in [None,population_size]:raise ValueError('Mixed populations need separate continuations')
        batch['population_size']=population_size
        if args.source_checkpoint:
            source_batch=BASE/'data'/args.source_experiment/'batch.json'
            policy=json.loads(source_batch.read_text())
            if not policy['options'].get('require_connected_core'):raise ValueError('Expected this campaign connected-core checkpoint')
            snapshot=directory/'checkpoint_inputs'/label.replace('/','__');snapshot.mkdir(parents=True)
            parent_layout=snapshot/'layout.npz';parent_trace=snapshot/'population32_trace.json'
            for source,target in [(parent/args.source_checkpoint,parent_layout),
                (parent/'population32_trace.json',parent_trace),(parent/'initial_metrics.json',snapshot/'initial_metrics.json'),
                (source_batch,snapshot/'source_batch.json')]:
                shutil.copy2(source,target);target.chmod(0o444)
            rounds=json.loads(parent_trace.read_text())
            report=dict(initial_counts=json.loads((snapshot/'initial_metrics.json').read_text())['counts'],
                frontier_search=dict(width=population_size,population_size=population_size,rounds=len(rounds),
                    configured_rounds=policy['rounds'],generated_trials=len(rounds)*population_size,
                    warm_start=False,count_penalty_used=False,source_running_checkpoint=True))
            parent_inputs=[parent_layout,parent_trace,snapshot/'initial_metrics.json',snapshot/'source_batch.json']
        else:
            report=json.loads((parent/'report.json').read_text())
            if not report.get('step4_contact_core_connected'):raise ValueError('Only current connected-core search slots can continue')
            parent_layout=parent/'layout.npz';parent_trace=parent/'population32_trace.json'
            parent_inputs=[parent/'report.json',parent_layout,parent_trace]
        out=base/'step4/step4.2'/args.experiment;out.mkdir(exist_ok=False);(out/'data').mkdir()
        original=read_upstream(base/'step4/step4.1/data/report.json')
        m=OracleModel(original['poses'],'B',demand_nodes=32,oracle_work_mode='normal');m.require_connected_core=True
        seed=base/'step3/step3.1/wrapped_support.obj';m.registered_seed=S.solid(trimesh.load(seed,force='mesh',process=False))
        m.inputs += [seed,*parent_inputs,controller]
        s=OracleSearch(m,out,iterations=0,frontier_width=population_size,search_policy='population_soft',seed=42)
        s.capture_process=False;s.baseline(m.evaluate(load_layout(base/'step4/step4.1/layout.npz')))
        evaluator=MaterialEvaluator(s);q=load_layout(parent_layout)
        def checked(q,operation,detail):
            result=m.evaluate(q);item=evaluator.preview(q,operation=operation,detail=detail,result=result)
            evaluator.certify_core(item);return item
        best=checked(q,'continued-retained-slot',dict(source=str(parent.relative_to(ROOT))))
        history=[];measurements=1
        if not failed_count(best.result):
            evaluator.connect(best);best=evaluator.construct(best)
            history.append(dict(operation='actual-core-and-Oracle-recertification',completed=best.completed,after_failed=failed_count(best.result)))
        for step in range(args.maximum_adjustments):
            if best.completed:break
            if not failed_count(best.result):
                evaluator.connect(best);best=evaluator.construct(best)
                if best.completed:break
            owner=max(best.layout.active,key=lambda k:np.count_nonzero(~best.result['masks'][k]))
            angular=tangent_frame(best.layout.directions[owner]);proposals=[]
            grounded=best.layout.copy();changed=[]
            for k in best.layout.active:
                if best.result['masks'][k].all()or registered(best.layout,k):continue
                height=m.workpiece_height(best.layout,k)
                if height<=1e-9:continue
                T=installed_state(m.native,best.layout,int(best.layout.hosts[k]))
                trial=shift(m,best.layout,k,T[:3,:3].T@np.array([0.,0.,-height]))
                proposals.append(('restore-ground-contact',trial,dict(owner=k,removed_airborne_height_m=height)))
                grounded=shift(m,grounded,k,T[:3,:3].T@np.array([0.,0.,-height]));changed.append(k)
            if len(changed)>1:proposals.insert(0,('restore-ground-contact-cohort',grounded,dict(owners=changed)))
            for degrees in [.03125,.125,.5,2.]:
                for axis in range(2):
                    for sign in [-1.,1.]:
                        proposals.append(('direction',turn(m,best.layout,owner,sign*np.radians(degrees)*angular[:,axis]),dict(owner=owner,degrees=sign*degrees,axis=axis)))
            if not registered(best.layout,owner):
                frame=translation_frame(m,best.layout,owner)
                for fraction in [1/2048,1/512,1/128]:
                    for axis in range(3):
                        for sign in [-1.,1.]:
                            proposals.append(('translation',shift(m,best.layout,owner,sign*m.extent*fraction*frame[:,axis]),dict(owner=owner,fraction=fraction,axis=axis,sign=sign)))
            before=failed_count(best.result);candidates=[best]
            for kind,trial,detail in proposals:
                try:
                    for k in trial.active:m.validate_placement(trial,k)
                    candidate=checked(trial,'retained-slot-'+kind,detail);measurements+=1;candidates.append(candidate)
                    if not failed_count(candidate.result)and candidate.oracle.get('enclosure_passed'):
                        evaluator.connect(candidate);candidate=evaluator.construct(candidate)
                        if candidate.completed:best=candidate;break
                except(ValueError,RuntimeError,AssertionError)as error:
                    history.append(dict(step=step,error=str(error),proposal=detail))
            if not best.completed:
                best=min(candidates,key=lambda x:(failed_count(x.result),not x.oracle.get('enclosure_passed'),float(x.oracle.get('gap_m',0.)),x.loss))
            history.append(dict(step=step,before_failed=before,after_failed=failed_count(best.result),selected=best.detail,completed=best.completed))
            print('CORE CONTINUATION',label,step,before,'->',failed_count(best.result),flush=True)
            if best.completed or failed_count(best.result)>=before:break
        initial=base/'step4/step4.1'/args.experiment
        shutil.copytree(base/'step4/step4.1'/args.source_experiment,initial)
        shutil.copy2(parent_trace,out/'population32_trace.json')
        save(out/'additional_adjustments.json',dict(source_experiment=args.source_experiment,source_checkpoint=args.source_checkpoint,
            history=history,actual_core_measurements=measurements,original_slots_reinitialized=False,additional_population_slots=0))
        s.material_selected=best;s.frontier_witnesses={best.layout.key():best.witness}if best.witness is not None else{}
        s.frontier_statistics=dict(report['frontier_search'],post_search_continuation=True,
            continuation_source_experiment=args.source_experiment,additional_adjustment_passes=len(history),
            additional_actual_core_measurements=measurements)
        best.result['base_oracle']=best.oracle;m.commit(best.result)
        saved=s.save(best.result,'connected_retained_slot_continuation',began,
            dict(source='original registered parent initialization',resume_own_experiment=args.source_experiment,
                historical_accepted_layouts_used=False,original_slots_reinitialized=False),initial_counts=report['initial_counts'])
        save(out/'report.json',saved);data=dict(saved);data['artifacts']={'../'+k:v for k,v in saved['artifacts'].items()};save(out/'data/report.json',data)
        result=dict(id=label,experiment=args.experiment,passed=saved['passed'],force_passed=saved['force_passed'],
            base_oracle_passed=saved['base_oracle_passed'],pose_count=len(original['poses']),counts=saved['counts'],
            frontier_width=population_size,frontier_search=saved['frontier_search'],rotating_reuse_pose_count=saved['rotating_reuse_pose_count'],
            juxtaposed_pose_count=saved['juxtaposed_pose_count'],solid_constructed_during_search=saved['solid_constructed_during_search'],
            oracle_status=saved['base_oracle']['status'],seconds=time.monotonic()-began,post_search_continuation=True)
        save(out/'run_result.json',result);A.publish_visuals(out);A.cleanup_stage(out)
        batch['results'].append(result);batch['passed']=sum(r['passed']for r in batch['results']);save(directory/'batch.json',batch)
    batch['complete']=True;save(directory/'batch.json',batch)


if __name__=='__main__':main()
