"""Fixed +Z comparison with identical free-seating construction protocol."""
import json,time,shutil
from pathlib import Path
import numpy as np
from codes.precompute_objects import head_cache as HC
from step3_scheculer import build_shared_fixture_v35 as B,contacts as I,operation_dsl as F
from step3_scheculer.initialize_current_v16 import current_task
from step5_current.evaluate import measure
from step3_scheculer.compare_shared_volume_v33 import GROUPS
from step3_scheculer.pair_scoring import J

STAGE='shared_fixture_plus_z_v36'

def baseline(group):
    base=I.OUTPUTS/'B'/group;source=base/'step3_scheculer/dsl_shared_direction_batch_v33/report.json';data=I.check_report(source)
    upsource=base/'step5_evaluate/shared_volume_comparison_v33/plus_z_contacts.json'
    up=json.loads(upsource.read_text());assert up['passed']
    tasks=[current_task('B',p) for p in data['poses']];groups=[]
    for pose,row in zip(data['poses'],up['poses']):
        cache=HC.load(HC.ROOT/'objects/B/poses'/pose,200,strict=True)
        byid={e['contact']['candidate_id']:e['contact'] for pool in cache['pools'].values() for e in pool}
        groups.append(tuple(byid[k] for k in row['selected_ids']))
    # Native heads are fixed across seating trials: certify original loads once.
    for task,contacts in zip(tasks,groups):
        fresh=[{k:v for k,v in c.items() if k not in ('wrench_generators','wrench_com_m')} for c in contacts]
        mask,_=J.classify(task.supply(fresh),task.targets)
        assert len(mask)==32768 and mask.all()
    seed=F.State(tuple(groups),np.repeat(np.eye(3)[None],len(tasks),axis=0),np.zeros((len(tasks),3)),tuple(F.ray([0,0,-1],0) for _ in tasks))
    out=base/'step4'/STAGE;out.mkdir(parents=True,exist_ok=True)
    for i,(label,state,cuts) in enumerate(B.proposals(group,tasks,seed)):
        folder=out/'attempts'/f'placement_{i:02d}'
        result=B.attempt(group,tasks,state,upsource,folder,label,cuts,True)
        if result is None:continue
        grow,report,mesh=result
        for name in ['shape.obj','geometry_certificate.npz','roots.obj','seed_targets.obj','partial_growth.obj','shared_tree.obj']:
            if (folder/name).exists():shutil.copy2(folder/name,out/name)
        F.save_state(out,tasks,state,dict(passed=True,all_original_loads_passed=True))
        aggregate,per_pose,_=measure([t.domain.mesh.vertices for t in tasks],mesh.vertices,state.bases,state.offsets)
        step5=base/'step5_evaluate'/STAGE;B.pictures(out,step5,tasks,state,mesh,grow,aggregate,group)
        final=dict(report,group=group,artifacts={n:I.sha256(out/n) for n in ['shape.obj','geometry_certificate.npz','overview.png','construction_steps.png']},provenance=dict(inputs=I.hashes([folder/'report.json',out/'state.json']),code=I.hashes([Path(__file__),Path(B.__file__)])))
        I.save(out/'report.json',final);I.check_report(out/'report.json')
        metric=dict(complete=True,passed=True,group=group,aggregate=aggregate,per_pose=per_pose,material_volume_cm3=report['volume_cm3'],provenance=dict(inputs=I.hashes([out/'report.json']),code=I.hashes([Path(__file__)])),artifacts={'overview.png':I.sha256(step5/'overview.png')})
        I.save(step5/'report.json',metric);return metric
    raise RuntimeError('No complete +Z comparison: '+group)

if __name__=='__main__':
    began=time.monotonic();rows=[]
    for group in GROUPS:
        stage=B.STAGE if group in ('pose1+3','pose3+6') else 'shared_fixture_seating_v34'
        selected_path=I.OUTPUTS/'B'/group/'step5_evaluate'/stage/'report.json';selected=I.check_report(selected_path)
        fixed=baseline(group) if group in ('pose1+3','pose3+6') else selected
        v=selected['aggregate']['object_and_support_poses']['box_volume_cm3'];z=fixed['aggregate']['object_and_support_poses']['box_volume_cm3']
        rows.append(dict(group=group,selected_volume_cm3=v,plus_z_volume_cm3=z,saved_percent=100*(1-v/z),baseline_same_solution=group not in ('pose1+3','pose3+6')))
        print('VOLUME',rows[-1],flush=True)
    out=I.OUTPUTS/'B'/GROUPS[-1]/'step5_evaluate'/B.STAGE
    I.save(out/'volume_comparison.json',dict(complete=True,passed=True,groups=rows,seconds=time.monotonic()-began,fixture_seating_free_in_both=True,global_optimality_claim=False))
