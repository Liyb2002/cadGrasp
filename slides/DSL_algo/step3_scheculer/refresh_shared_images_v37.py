"""Add actual exit arrows, required bearing hulls and XYZ volume labels.
Presentation only; no construction, geometry edits or acceptance replay.
"""
import json
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial import ConvexHull
from step3_scheculer import contacts as I,build_shared_fixture_v34 as B
from step3_scheculer.compare_shared_volume_v33 import GROUPS
from step3_scheculer.initialize_current_v16 import current_task
from step0_pose_selection.floor_points import pressure_centers
from step4_connect_support import clean_render as V


def xy(points,camera,size):
    focus,basis,span=camera;p=(np.asarray(points)-focus)@basis.T
    return np.c_[size/2+p[:,0]*size/span,size/2-p[:,1]*size/span]


def arrow(im,point,direction,camera,size,length):
    a,b=xy([point,point+length*np.asarray(direction)],camera,size)
    draw=ImageDraw.Draw(im);draw.line([tuple(a),tuple(b)],fill='#8b4da4',width=5)
    delta=b-a;n=np.linalg.norm(delta)
    if n>1:
        u=delta/n;v=np.array([-u[1],u[0]])
        draw.polygon([tuple(b),tuple(b-16*u+7*v),tuple(b-16*u-7*v)],fill='#8b4da4')
    label='Exit +Z' if abs(direction[2])>.999 else 'Exit +Y'
    draw.text(tuple(b+[5,5]),label,font=ImageFont.truetype(B.FONT,18),fill='#8b4da4')


def refresh(group,stage):
    base=I.OUTPUTS/'B'/group;out=base/'step4'/stage;step5=base/'step5_evaluate'/stage
    report=I.check_report(out/'report.json');metric=I.check_report(step5/'report.json');state=json.loads((out/'state.json').read_text())
    tasks=[current_task('B',p) for p in state['poses']];bases=np.asarray(state['placement']['bases']);offsets=np.asarray(state['placement']['offsets']);mesh=trimesh.load(out/'shape.obj',force='mesh',process=False)
    contacts=tuple(tuple(I.read_contacts(out/f'contacts_{t.pose}.npz')) for t in tasks)
    native_state=B.F.State(contacts,bases,offsets,tuple(state['exit_paths']))
    B.pictures(out,step5,tasks,native_state,mesh,None,metric['aggregate'],group)
    # Match the existing overview cameras exactly; add native-world exits.
    p=out/'overview.png';canvas=Image.open(p).convert('RGB');size=700;columns=min(3,len(tasks)+1)
    for i,(task,b,o,path) in enumerate(zip(tasks,bases,offsets,state['exit_paths'])):
        installed=(mesh.vertices-o)@b.T;direction=np.asarray(path['initial_object_exit_world'])
        camera=B.PUB.CPU.fit(np.vstack([installed,task.domain.mesh.vertices]),direction+[.5,-.8,.6],1.,padding=1.3)
        panel=canvas.crop(((i%columns)*size,(i//columns)*size,(i%columns+1)*size,(i//columns+1)*size))
        center=task.domain.mesh.vertices.mean(0);arrow(panel,center,direction,camera,size,float(task.domain.mesh.extents.max())*.6)
        canvas.paste(panel,((i%columns)*size,(i//columns)*size))
    canvas.save(p)
    # Required ground hulls are constraints, not preselected physical feet.
    sheet=Image.open(out/'construction_steps.png').convert('RGB')
    cloud=np.vstack([mesh.vertices,tasks[0].domain.mesh.vertices@bases[0]+offsets[0]])
    camera=B.PUB.CPU.fit(cloud,[1.,-1.3,.8],1.,padding=1.3)
    panel=sheet.crop((2*size,0,3*size,size));draw=ImageDraw.Draw(panel)
    for task,b,o in zip(tasks,bases,offsets):
        required=pressure_centers(task.targets/task.scale,task.domain.com)[0];hull=ConvexHull(required)
        poly=np.c_[required[hull.vertices],np.zeros(len(hull.vertices))]@b+o
        pixels=xy(poly,camera,size);draw.line([tuple(v) for v in pixels]+[tuple(pixels[0])],fill='#2d89bb',width=3)
    draw.text((20,size-35),'Blue: required bearing coverage',font=ImageFont.truetype(B.FONT,19),fill='#2d89bb');sheet.paste(panel,(2*size,0))
    panel=sheet.crop((3*size,0,4*size,size));center=tasks[0].domain.mesh.vertices.mean(0)@bases[0]+offsets[0]
    d=np.asarray(state['exit_paths'][0]['initial_object_exit_world']);arrow(panel,center,d@bases[0],camera,size,float(tasks[0].domain.mesh.extents.max())*.6)
    sheet.paste(panel,(3*size,0));sheet.save(out/'construction_steps.png')
    # The selected Step5 image includes the measured XYZ volume, not only XY area.
    p=step5/'overview.png';im=Image.open(p).convert('RGB');expanded=Image.new('RGB',(im.width,im.height+75),'white');expanded.paste(im,(0,0))
    a=metric['aggregate'];label=f"XYZ box volume: objects {a['object_poses']['box_volume_cm3']:.1f} cm3 | with support {a['object_and_support_poses']['box_volume_cm3']:.1f} cm3"
    ImageDraw.Draw(expanded).text((im.width//2,im.height+20),label,anchor='mt',font=ImageFont.truetype(B.FONT,27),fill='#344753');expanded.save(p)
    report['presentation_refresh']=dict(geometry_changed=False,exit_arrows=True,required_bearing_hulls=True)
    report['provenance']['code'].update(I.hashes([Path(__file__)]))
    for name in ('overview.png','construction_steps.png'):report['artifacts'][name]=I.sha256(out/name)
    I.save(out/'report.json',report);I.check_report(out/'report.json')
    metric['provenance']['inputs'].update(I.hashes([out/'report.json']))
    metric['provenance']['code'].update(I.hashes([Path(__file__)]));metric['artifacts']['overview.png']=I.sha256(p)
    I.save(step5/'report.json',metric);I.check_report(step5/'report.json')

if __name__=='__main__':
    for group in GROUPS:refresh(group,'shared_fixture_seating_v35' if group in ('pose1+3','pose3+6') else 'shared_fixture_seating_v34')
