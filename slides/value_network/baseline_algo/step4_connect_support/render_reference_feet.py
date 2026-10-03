"""Image-only reference presentation: readable workpiece and compact pale feet."""
import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step4_connect_support import reseating as R,clean_render as V,publish_compact as P

SUPPORT='#cbd2ce'
OBJECT='#acb9c0'
OPACITY=.88
VIEW=[.68,-1.,.8]


def smooth_object(task):
    mesh=task.domain.mesh
    colors=np.tile(P.CPU.rgb(OBJECT),(len(mesh.faces),3,1))
    colors[task.domain.work_ids]=P.CPU.rgb('#b4cba8')
    part=P.CPU.piece(dict(v=mesh.vertices.ravel(),f=mesh.faces.ravel()),colors,smooth=True)
    part['opacity']=OPACITY
    return part


def render(group,candidate=None):
    out=group/'step4'
    report_path=out/'data/report.json' if candidate is None else candidate/'outer_report.json'
    report=I.check_report(report_path);case=R.load_case(out)
    work=out/report['body_directory'];body_path=work/'fixture.obj'
    body=trimesh.load(body_path,force='mesh',process=False)
    destination=out if candidate is None else candidate/'preview';destination.mkdir(exist_ok=True)
    bases=np.asarray(report['placement']['bases']);offsets=np.asarray(report['placement']['offsets'])
    heads=[]
    for k,contacts in enumerate(case.groups):
        for c in contacts:
            points=c['triangles_m'].reshape(-1,3)@bases[k]+offsets[k]
            mesh=trimesh.Trimesh(points,np.arange(len(points)).reshape(-1,3),process=False)
            heads.append((c['candidate_id'],mesh))
    colors=V.head_colors([ident for ident,_ in heads])
    overlays=[(ident,V.collar(body,m.vertices.min(0)-.004,m.vertices.max(0)+.004)) for ident,m in heads]
    support=[P.piece(body,SUPPORT)]
    support.extend(P.piece(m,colors[ident],.000003) for ident,m in heads)
    support.extend(P.piece(m,colors[ident],.000002) for ident,m in overlays if len(m.faces))
    renderer=V.Renderer();pictures=[];rows=[];artifacts={}
    for k,task in enumerate(case.tasks):
        b,o=bases[k],offsets[k];translation=-b@o
        vertices=body.vertices@b.T+translation
        points=np.vstack([vertices,task.domain.mesh.vertices])
        low,high=points.min(0),points.max(0)
        floor=P.CPU.box(np.r_[high[:2]-low[:2]+.035,.001],np.r_[(low[:2]+high[:2])/2,-.0006],'#eff3f4');floor['unlit']=True
        parts=[P.CPU.placed(floor)]+[P.CPU.placed(p,b,translation) for p in support]
        parts.append(P.CPU.placed(smooth_object(task)))
        camera=P.CPU.fit(points,VIEW,1.,padding=1.25)
        picture=renderer.render(parts,camera,1100);filename=f'{task.pose}.png';picture.save(destination/filename)
        pictures.append(picture);artifacts[filename]=I.sha256(destination/filename)
        rows.append(dict(pose=task.pose,image=filename,fixture_rotation_world=b.tolist(),
            fixture_translation_world=translation.tolist(),camera=dict(focus_world=camera[0].tolist(),
                basis_world=camera[1].tolist(),span_m=float(camera[2]))))
    camera=P.CPU.fit(body.vertices,VIEW,1.,padding=1.25)
    picture=renderer.render([P.CPU.placed(p) for p in support],camera,1100)
    picture.save(destination/'support.png');pictures.append(picture)
    canvas=Image.new('RGB',(1100*len(pictures),1100),'white')
    for j,picture in enumerate(pictures):canvas.paste(picture,(1100*j,0))
    canvas.save(destination/'overview.png')
    for filename in ('support.png','overview.png'):artifacts[filename]=I.sha256(destination/filename)
    metadata=dict(complete=True,poses=case.poses,per_pose=rows,object_opacity=OPACITY,
        support_color=SUPPORT,head_colors=colors,text_in_images=False,arrows_in_images=False,
        object_smooth_normals=True,reference='slides/baseline_algo/output/B/refer.png',
        support_image='support.png',overview='overview.png',overview_panels=[r['image'] for r in rows]+['support.png'],
        image_size=[1100,1100],combined_size=list(canvas.size),html_generated=False,videos_generated=False,
        provenance=dict(inputs=I.hashes([report_path,body_path,I.OUTPUTS/'B/refer.png']),
            code=I.hashes([Path(__file__),Path(V.__file__),Path(P.CPU.__file__)])),
        artifacts={('../'+name if candidate is None else name):sha for name,sha in artifacts.items()})
    I.save(destination/'visualization.json' if candidate is not None else out/'data/visualization.json',metadata)
    print('REFERENCE PICTURES',group.name,str(destination/'overview.png'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('group');p.add_argument('--candidate',type=Path)
    args=p.parse_args();render(I.OUTPUTS/'B'/args.group,args.candidate)
