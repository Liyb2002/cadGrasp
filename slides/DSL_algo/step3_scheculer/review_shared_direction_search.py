"""Fresh original-equation checks for final and nonvertical shared exits."""
import json,time
from pathlib import Path
import numpy as np
from codes.precompute_objects import head_cache as HC
from step3_scheculer import contacts as I,shared_direction_search as D
from step3_scheculer.initialize_current_v16 import current_task
from step3_scheculer.pair_scoring import J


def audit(name,set_id):
    start=time.monotonic()
    out=I.OUTPUTS/name/set_id/'step3_scheculer'/D.STAGE
    source=out/'report.json';report=I.check_report(source)
    assert report['contact_programs_passed'] and not report['object_poses_changed']
    assert report['world_xyz_frame'] and not report['exit_direction_prespecified']
    assert not report['full_support_constructed']
    alternatives=[r for r in report['history'] if r['passed'] and r.get('operation')
                  and np.linalg.norm(np.asarray(r['direction_world'])-[0,0,1])>1e-5]
    states=[('final',report['shared_object_exit_world'],report['per_pose'])]
    if alternatives:
        r=alternatives[0];states.append(('nonvertical',r['direction_world'],r['poses']))
    checked=[];inputs=[source];artifacts={}
    for label,direction,rows in states:
        assert len(rows)==len(report['poses'])
        poses=[]
        for pose,row in zip(report['poses'],rows):
            assert row['pose']==pose and row['passed']
            task=current_task(name,pose)
            cache=HC.load(I.ROOT/'objects'/name/'poses'/pose,strict=True)
            entries={e['contact']['candidate_id']:e for pool in cache['pools'].values() for e in pool}
            selected=[entries[k] for k in row['selected_ids']]
            contacts=[]
            for e in selected:
                assert e['valid'] and e['local_clearance']['valid']
                c=e['contact'];points=c['triangles_m'].reshape(-1,3)
                normals=np.repeat(-task.domain.mesh.face_normals[c['source_faces']],3,axis=0)
                np.testing.assert_allclose(np.c_[normals,np.cross(points-task.domain.com,normals)],c['wrench_generators'],atol=1e-12,rtol=0)
                assert not np.intersect1d(c['source_faces'],task.domain.work_ids).size
                contacts.append({k:v for k,v in c.items() if k not in ('wrench_generators','wrench_com_m')})
            mask,info=J.classify(task.supply(contacts),task.targets)
            assert len(mask)==32768 and mask.all()
            np.testing.assert_allclose(row['path']['initial_object_exit_world'],direction,atol=1e-12)
            analyzer=D.P.PathAnalyzer(task.domain.mesh,cache['report']['depth_m'])
            path=analyzer.test(analyzer.heads(contacts),row['path'])
            assert path['clear']
            coverage=out/f'audit_{label}_{pose}.npz';np.savez_compressed(coverage,mask=mask)
            artifacts[coverage.name]=I.sha256(coverage)
            poses.append(dict(pose=pose,heads=len(selected),covered=int(mask.sum()),cpu_check=info,path_check=path))
            inputs+=task.inputs
        checked.append(dict(label=label,direction_world=direction,per_pose=poses))
    result=dict(complete=True,passed=True,scope=report['passed_scope'],full_support_constructed=False,
        checked=checked,seconds=time.monotonic()-start,
        provenance=dict(inputs=I.hashes(inputs),code=I.hashes([Path(__file__),Path(D.__file__)]+D.P.sources())),artifacts=artifacts)
    I.save(out/'audit.json',result);I.check_report(out/'audit.json')
    print(set_id,'AUDIT PASSED',[(r['label'],r['direction_world']) for r in checked],flush=True)
    return result

if __name__=='__main__':
    for group in ('pose3+15','pose1+12+29','pose8+9+13+30'):audit('B',group)
