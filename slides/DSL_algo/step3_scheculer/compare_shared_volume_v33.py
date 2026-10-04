"""Run current-dataset contact search results through full construction and Step5.
Historical outputs are preserved. Identity fixture registration keeps native poses fixed.
"""
import argparse,json,time,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import trimesh
from codes.precompute_objects import head_cache as HC
from step3_scheculer import contacts as I,shared_direction_batch_v33 as Q
from step3_scheculer import shared_direction_paths as P,operation_dsl as F
from step3_scheculer.operation_growth_recovery import RecoveryGrow
from step3_scheculer.initialize_current_v16 import current_task
from step3_scheculer.pair_scoring import J
from step5_current.evaluate import measure

STAGE='shared_volume_comparison_v33'
GROUPS=['pose1+3','pose1+2+3+4+5','pose1+2+8+17','pose2+10+15','pose2+12+15','pose2+9+13+15+17','pose3+6','pose5+7','pose6+8+10+19']


def construct(group,tasks,caches,trial,out,source):
    started=time.monotonic();out.mkdir(parents=True,exist_ok=True)
    direction=np.asarray(trial['direction_world']);contacts=[]
    result=dict(complete=True,passed=False,direction_world=direction.tolist(),seconds=0.,stage='contact_coverage',volume_cm3=None)
    try:
        for task,cache,row in zip(tasks,caches,trial['poses']):
            byid={e['contact']['candidate_id']:e['contact'] for pool in cache['pools'].values() for e in pool}
            cs=[byid[k] for k in row['selected_ids']]
            contacts.append(cs)
            I.save_contacts(out/f'contacts_{task.pose}.npz',cs)
        result['stage']='full_shared_fixture_setup'
        state=F.State(tuple(tuple(cs) for cs in contacts),np.repeat(np.eye(3)[None],len(tasks),axis=0),np.zeros((len(tasks),3)),tuple(F.ray(-direction,0) for _ in tasks))
        # The guide is only a navigation input, never an accepted prior fixture.
        vertices=np.vstack([t.domain.mesh.vertices for t in tasks]);lo=vertices.min(0)-.05;hi=vertices.max(0)+.05;lo[2]=0.
        guide=trimesh.creation.box(extents=hi-lo,transform=trimesh.transformations.translation_matrix((hi+lo)/2))
        guide_path=out/'navigation_guide.obj';F.SPACE.export_exact_obj(guide,guide_path)
        ref=dict(complete=True,passed=False,scope='navigation guide only; no prior accepted fixture',construction={},provenance=dict(inputs={},code={}))
        previous=F.E.PathAnalyzer;F.E.PathAnalyzer=P.PathAnalyzer
        try:
            grow=RecoveryGrow(SimpleNamespace(name=group),tasks,state,out,guide_path,ref)
            for task,cs in zip(tasks,contacts):
                fresh=[{k:v for k,v in c.items() if k not in ('wrench_generators','wrench_com_m')} for c in cs]
                mask,_=J.classify(task.supply(fresh),task.targets)
                if not mask.all():raise RuntimeError('Full original-load CPU coverage failed: '+task.pose)
            grow.case.paths.append(source);grow.recheck_export=False
            result['stage']='full_shared_fixture_growth'
            report=grow.run(.008)
        finally:F.E.PathAnalyzer=previous
        # Exact Step5 metric from the constructed in-memory solid's exported vertices;
        # this is measurement, not a second acceptance or replay.
        mesh=trimesh.load(out/'shape.obj',force='mesh',process=False)
        aggregate,per_pose,_=measure([t.domain.mesh.vertices for t in tasks],mesh.vertices,state.bases,state.offsets)
        report.pop('material_reduction_percent',None);report.pop('previous_material_volume_cm3',None)
        report['construction']['reference_used_as']='unaccepted navigation guide; no inherited ground targets'
        report['provenance']['code'].update(I.hashes([Path(__file__),Path(F.__file__),Path(P.__file__),Path(__file__).with_name('operation_growth_recovery.py')]))
        I.save(out/'report.json',report)
        result.update(passed=True,stage='step5_measured',volume_cm3=aggregate['object_and_support_poses']['box_volume_cm3'],material_cm3=report['volume_cm3'],aggregate=aggregate,per_pose=per_pose,heads=[len(x) for x in contacts],validation_policy=report['validation_policy'])
    except Exception as error:
        result.update(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        if hasattr(error,'access_report'):result['access_report']=error.access_report
    result['seconds']=time.monotonic()-started
    result['provenance']=dict(inputs=I.hashes([source]+[p for t in tasks for p in t.inputs]),code=I.hashes([Path(__file__)]))
    I.save(out/'comparison.json',result)
    print('CONSTRUCTION',group,np.round(direction,4),result['passed'],result.get('error',result.get('volume_cm3')),flush=True)
    return result


def run(group):
    started=time.monotonic();base=I.OUTPUTS/'B'/group
    out=base/'step5_evaluate'/STAGE;out.mkdir(parents=True,exist_ok=True)
    source=base/'step3_scheculer'/Q.STAGE/'report.json'
    data=I.check_report(source)
    tasks=[current_task('B',pose) for pose in data['poses']]
    caches=[HC.load(HC.ROOT/'objects/B/poses'/pose,200,strict=True) for pose in data['poses']]
    trials=[t for t in data['history'] if t['passed']]
    up=next((t for t in trials if np.linalg.norm(np.asarray(t['direction_world'])-[0,0,1])<1e-8),None)
    if up is None:
        searches=[Q.Search('B',pose,'cuda') for pose in data['poses']];rows=[]
        with Q.bounded_lps():
            for search in searches:
                _,row=search.solve_direction(np.array([0.,0.,1.]),data['config']['max_heads'],data['config']['shortlist']);rows.append(row)
        up=dict(direction_world=[0.,0.,1.],passed=all(r['passed'] for r in rows),poses=rows)
        I.save(out/'plus_z_contacts.json',up)
        if up['passed']:trials.append(up)
        del searches
    if up['passed']:baseline=construct(group,tasks,caches,up,base/'step4'/STAGE/'plus_z',source)
    else:baseline=dict(passed=False,stage='contact_search',error='Bounded +Z head search failed',volume_cm3=None)
    candidates=[];seen={Q.direction_key([0,0,1])}
    # Record selected algorithm output first, then its feasible alternatives.
    trials.sort(key=lambda t:np.linalg.norm(np.asarray(t['direction_world'])-np.asarray(data['shared_object_exit_world'] or [0,0,1])))
    for i,trial in enumerate(trials):
        key=Q.direction_key(trial['direction_world'])
        if key in seen:continue
        seen.add(key)
        candidates.append(construct(group,tasks,caches,trial,base/'step4'/STAGE/f'direction_{i:02d}',source))
    selected=baseline if np.linalg.norm(np.asarray(data['shared_object_exit_world'] or [99,99,99])-[0,0,1])<1e-8 else next((r for r in candidates if Q.direction_key(r['direction_world'])==Q.direction_key(data['shared_object_exit_world'])),None)
    qualified=[r for r in [baseline]+candidates if r['passed']]
    best=min(qualified,key=lambda r:r['volume_cm3']) if qualified else None
    result=dict(complete=True,group=group,poses=data['poses'],current_dataset=True,object_poses_changed=False,registration='identity; native fixed world poses',contact_search_direction=data['shared_object_exit_world'],contact_heads=[r['heads'] for r in data['per_pose']],contact_search_seconds=data['seconds'],plus_z=baseline,algorithm_selected=selected,candidates=candidates,best_constructed=best,actual_volume_improvement_percent=(100*(1-best['volume_cm3']/baseline['volume_cm3']) if best and baseline['passed'] else None),seconds=time.monotonic()-started,provenance=dict(inputs=I.hashes([source]),code=I.hashes([Path(__file__)])))
    I.save(out/'report.json',result);return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--sets',nargs='+',default=GROUPS);args=parser.parse_args()
    results=[run(g) for g in args.sets]
    I.save(I.OUTPUTS/'B'/args.sets[-1]/'step5_evaluate'/STAGE/'batch.json',dict(complete=True,groups=results))
