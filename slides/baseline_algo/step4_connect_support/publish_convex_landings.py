"""Static placement pictures for all independently seated pose combinations."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step4_connect_support import reseating as R
from step4_connect_support import publish_compact as P
from step4_connect_support import clean_render as V
from step4_connect_support import presentation_camera as C
from step4_connect_support.run_reseated import groups

def publish(group):
    output=group/'step4';path=output/'data/report.json';report=I.check_report(path)
    case=R.load_case(output);placement=report.get('placement')
    if placement is None:raise ValueError('No placement available to illustrate')
    bases=np.asarray(placement['bases']);offsets=np.asarray(placement['offsets'])
    body=trimesh.load(output/'shape.obj',force='mesh',process=False) if report['constructed'] else None
    heads=[]
    for k,contacts in enumerate(case.groups):
        for contact in contacts:
            points=contact['triangles_m'].reshape(-1,3)@bases[k]+offsets[k]
            mesh=trimesh.Trimesh(points,np.arange(len(points)).reshape(-1,3),process=False)
            heads.append((k,contact['candidate_id'],mesh))
    colors=V.head_colors([ident for _,ident,_ in heads])
    overlays=[]
    if body is not None:
        for _,ident,m in heads:
            overlays.append((ident,V.collar(body,m.vertices.min(0)-.004,m.vertices.max(0)+.004)))
    renderer=V.Renderer();rows=[];pictures=[];artifacts={}
    for k,task in enumerate(case.tasks):
        b,o=bases[k],offsets[k];translation=-b@o
        static=[]
        if body is not None:
            support=body.vertices@b.T+translation
            static.append(P.CPU.placed(P.piece(body,V.SUPPORT),b,translation))
        else:support=np.vstack([m.vertices for _,_,m in heads])@b.T+translation
        points=np.vstack([support,task.domain.mesh.vertices])
        floor=P.CPU.box(np.r_[np.ptp(points[:,:2],axis=0)+.04,.001],
            np.r_[(points[:,:2].min(0)+points[:,:2].max(0))/2,-.0006],'#f0f3f5');floor['unlit']=True
        obj=P.piece(task.domain.mesh,V.OBJECT);obj['opacity']=V.OPACITY
        parts=[P.CPU.placed(floor)]+static+[P.CPU.placed(obj)]
        parts += [P.CPU.placed(P.piece(m,colors[ident],.000003),b,translation) for _,ident,m in heads]
        parts += [P.CPU.placed(P.piece(m,colors[ident],.000002),b,translation) for ident,m in overlays if len(m.faces)]
        world_body=trimesh.Trimesh(support,body.faces,process=False)
        active={ident for owner,ident,_ in heads if owner==k}
        active_meshes=[trimesh.Trimesh(m.vertices@b.T+translation,m.faces,process=False)
            for ident,m in overlays if ident in active and len(m.faces)]
        camera,camera_info=C.choose(renderer,world_body,active_meshes,task.domain.mesh)
        page=renderer.render(parts,camera,1000)
        direction=-np.asarray(placement['directions'][k])
        if body is not None:
            check=report['construction']['checks'][k]
            label=('退出通过' if check['withdrawal']['clear'] else '退出未通过')+' · '+('原载荷全部通过' if check['coupled_equilibrium_passed'] else '承载验收未通过')
        else:label='仅接触组摆放候选 · 尚无连接实体'
        filename=f'{task.pose}.png';page.save(output/filename)
        rows.append(dict(pose=task.pose,image=filename,constructed=body is not None,
            status=label,step3_covered=case.schedule['covered_counts'][k],
            fixture_rotation_world=b.tolist(),fixture_translation_world=translation.tolist(),
            object_exit_direction_world=direction.tolist(),camera=camera_info))
        pictures.append(page);artifacts['../'+filename]=I.sha256(output/filename)
        print('POSE RENDERED',group.name,task.pose,'front fraction',
            round(camera_info['previous_fixed_view']['support_front_fraction'],3),'->',
            round(camera_info['selected']['support_front_fraction'],3),flush=True)
    support_camera,support_camera_info=C.choose(renderer,body,[m for _,m in overlays])
    support_parts=[P.CPU.placed(P.piece(body,V.SUPPORT))]
    support_parts += [P.CPU.placed(P.piece(m,colors[ident],.000003)) for _,ident,m in heads]
    support_parts += [P.CPU.placed(P.piece(m,colors[ident],.000002)) for ident,m in overlays if len(m.faces)]
    support_image=renderer.render(support_parts,support_camera,1000)
    support_image.save(output/'support.png');pictures.append(support_image)
    artifacts['../support.png']=I.sha256(output/'support.png')
    columns=2 if len(pictures)==4 else min(3,len(pictures))
    height=1000*math.ceil(len(pictures)/columns)
    canvas=Image.new('RGB',(1000*columns,height),'white')
    for k,picture in enumerate(pictures):
        row=k//columns
        count=min(columns,len(pictures)-row*columns)
        left=(columns-count)*500
        canvas.paste(picture,(left+(k%columns)*1000,row*1000))
    canvas.save(output/'overview.png');artifacts['../overview.png']=I.sha256(output/'overview.png')
    I.save(output/'data/visualization.json',dict(complete=True,poses=case.poses,per_pose=rows,
        image_size=[1000,1000],text_in_images=False,arrows_in_images=False,
        object_opacity=V.OPACITY,support_color=V.SUPPORT,head_colors=colors,combined_size=list(canvas.size),videos_generated=False,
        overview='overview.png',overview_panels=[r['image'] for r in rows]+['support.png'],
        support_image='support.png',support_camera=support_camera_info,physical_geometry_unchanged=True,html_generated=False,
        provenance=dict(inputs=I.hashes([path]+([output/'shape.obj'] if body is not None else [])),
            code=I.hashes([Path(__file__),Path(C.__file__),Path(P.CPU.__file__),Path(P.__file__),Path(V.__file__),Path(V.__file__).with_name("translucent_raster.cpp")])),artifacts=artifacts))
    for old in [output/'placements.png',*output.glob('placement_pose_*.png')]:old.unlink(missing_ok=True)
    print('PLACEMENTS PUBLISHED',group.name,len(rows),'poses',flush=True)


def gallery(name):
    """Print the image index; keep B-level files free of batch JSON/pages."""
    base=I.OUTPUTS/name;rows=[]
    for group in groups(name):
        output=group/'step4';report=I.check_report(output/'data/report.json')
        visual=I.check_report(output/'data/visualization.json')
        rows.append(dict(group=group.name,poses=report['poses'],constructed=report['constructed'],passed=report['passed'],
            status=report['status'],volume_cm3=report.get('volume_cm3'),images=len(visual['per_pose']),
            overview=f'{group.name}/step4/overview.png',support=f'{group.name}/step4/support.png'))
    print(json.dumps(dict(complete=True,groups=rows,videos_generated=False,html_generated=False),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('object',nargs='?',default='B');p.add_argument('--group')
    args=p.parse_args()
    for group in ([I.OUTPUTS/args.object/args.group] if args.group else groups(args.object)):publish(group)
    if not args.group:gallery(args.object)
