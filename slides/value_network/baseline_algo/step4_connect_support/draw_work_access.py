"""Text-free actual counterexamples: working area, obstructing body and ray."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image,ImageDraw
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step3_scheculer.pair_tasks import read_task
from step4_connect_support import clean_render as V,publish_compact as P
from step4_connect_support.surface_check import winding_number


def draw(group):
    out=group/'step4';data=out/'data'
    report=I.check_report(data/'report.json');audit=I.check_report(data/'work_access_check.json')
    mesh=trimesh.load(out/'shape.obj',force='mesh',process=False)
    work=out/report['body_directory'];design=json.loads((work/'design.json').read_text())
    paths=[data/'report.json',data/'work_access_check.json',out/'shape.obj',work/'design.json']
    renderer=V.Renderer();pictures=[];rows=[]
    for k,row in enumerate(audit['checks']):
        w=row['original_access_counterexample']
        if not w:continue
        pose=row['pose'];task=read_task(group.parent.name,pose,
            folder=group/'step3_scheculer/independent_poses_floor2mm'/pose/'step_1_needs');paths+=task.inputs
        b=np.asarray(report['placement']['bases'][k]);o=np.asarray(report['placement']['offsets'][k]);translation=-b@o
        point=np.asarray(w['interior_probe_world_m'])@b+o
        parts=[P.CPU.placed(P.piece(mesh,'#c5ccc9'),b,translation)];owners=[]
        for record in design['local_bodies']:
            path=work/record['file'];part=trimesh.load(path,force='mesh',process=False)
            if abs(winding_number(part,point))>.99:
                parts.append(P.CPU.placed(P.piece(part,'#d76f64',.000003),b,translation))
                owners.append(dict(candidate_id=record['candidate_id'],head_pose=record['head_pose'],file=record['file']))
                paths.append(path)
        colors=np.tile(P.CPU.rgb('#b3bec4'),(len(task.domain.mesh.faces),3,1))
        colors[task.domain.work_ids]=P.CPU.rgb('#76ae70')
        obj=P.CPU.piece(dict(v=task.domain.mesh.vertices.ravel(),f=task.domain.mesh.faces.ravel()),colors,smooth=True)
        obj['opacity']=.55;parts.append(P.CPU.placed(obj))
        q=np.asarray(w['work_point_world_m']);u=np.asarray(w['outward_direction_world'])
        hit=np.asarray(w['hit_point_world_m']);end=q+u*(w['ray_hit_distance_m']+.025)
        view=np.cross(u,[0,0,1])+.35*u+[0,0,.8]
        points=np.vstack([(mesh.vertices-o)@b.T,task.domain.mesh.vertices,q,end])
        camera=P.CPU.fit(points,view,1.,padding=1.15)
        size=1200;picture=renderer.render(parts,camera,size)
        focus,basis,span=camera
        p=(np.array([q,hit,end])-focus)@basis.T
        xy=np.c_[size/2+p[:,0]*size/span,size/2-p[:,1]*size/span]
        # The ray is an X-ray annotation at its exact projected coordinates.
        # It is not added to the physical mesh, and has no arrowhead or label.
        ink=ImageDraw.Draw(picture);ink.line([tuple(xy[0]),tuple(xy[2])],fill='#b83837',width=3)
        for center,color in [(xy[0],'#42833c'),(xy[1],'#bd3232')]:
            x,y=center;ink.ellipse([x-6,y-6,x+6,y+6],fill=color,outline='white',width=2)
        pictures.append(picture);rows.append(dict(pose=pose,obstructing_local_bodies=owners,
            witness=w,camera=dict(focus_world=focus.tolist(),basis_world=basis.tolist(),span_m=span)))
    canvas=Image.new('RGB',(1200*len(pictures),1200),'white')
    for k,picture in enumerate(pictures):canvas.paste(picture,(1200*k,0))
    filename='work_access_witnesses.png';canvas.save(data/filename)
    I.save(data/'work_access_witnesses.json',dict(complete=True,checks=rows,
        red_body='Actual local body containing the independent interior witness; the whole red body is not claimed to obstruct every ray',
        green_surface='Original task working surface',red_line='Exact projected original processing half-ray segment; X-ray overlay',
        text_in_images=False,arrows_in_images=False,physical_mesh_unchanged=True,html_generated=False,
        provenance=dict(inputs=I.hashes(paths),code=I.hashes([Path(__file__),Path(V.__file__),Path(P.CPU.__file__)])),
        artifacts={filename:I.sha256(data/filename)}))
    print('ACCESS WITNESSES',data/filename,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('group',nargs='?',default='pose5+7')
    args=p.parse_args();draw(I.OUTPUTS/'B'/args.group)
