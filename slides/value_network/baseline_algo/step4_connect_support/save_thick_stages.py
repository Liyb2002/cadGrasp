"""Save actual per-pose context and cumulative meshes for the eight new supports."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step4_connect_support import boxed_support as F,deterministic_space as D
from step4_connect_support.run_thick_batch import GROUPS

def save(name):
    group=I.OUTPUTS/'B'/name;out=group/'step4/data/growing_support'
    report=I.check_report(out/'report.json')
    if 'minimum_branch_thickness' not in report:return
    case,_=F.read_case(group,out)
    b=np.asarray(report['placement']['bases']);o=np.asarray(report['placement']['offsets']);d=np.asarray(report['placement']['directions'])
    objects=[]
    for k,task in enumerate(case.tasks):
        mesh=task.domain.mesh.copy();mesh.vertices=mesh.vertices@b[k]+o[k]
        filename='object_'+task.pose+'_fixture.obj';D.export_exact_obj(mesh,out/filename)
        objects.append(dict(pose=task.pose,mesh=filename))
    boxed=I.check_report(group/'step4/data/boxed_support/report.json')
    I.save(out/'stages.json',dict(placement=report['placement'],poses=case.poses,
        reference_pose=case.poses[0],object_mesh=objects[0]['mesh'],objects=objects,coordinate_frame='fixture',
        camera_view_fixture=((-d[0]+np.array([0,0,.65]))@b[0]).tolist(),
        ideal_space_budget=boxed['ideal_box'],accepted_space_budget=report['result']['box'],actual_space_budget=report['space_budget'],
        stages=[dict(key=k,label=label,mesh=k+'.obj') for k,label in [('roots','Original contact roots'),('soles','Five-mm ground soles'),('local_bodies','Thick local connections'),('shared_tree','Direct shared rods')]]+
        [dict(key='final',label='Verified final support',mesh='shape.obj')]))
    print('ACTUAL STAGES SAVED',name,flush=True)
if __name__=='__main__':
    for name in sys.argv[1:] or GROUPS:save(name)
