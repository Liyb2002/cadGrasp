"""Static placement pictures for all independently seated pose combinations."""
import argparse
import math
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image,ImageDraw
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step4_connect_support.baseline_current import reseating as R
from step4_connect_support.baseline_current import publish_compact as P
from step4_connect_support.baseline_current.codesign_port.fixture_view import export_viewer
from step4_connect_support.baseline_current.run_reseated import groups

STYLE='<style>body{font:17px system-ui;max-width:1500px;margin:30px auto;padding:0 20px;color:#294553;background:#f5f8fa}p{line-height:1.6}img{max-width:100%}section{background:white;padding:18px;margin:20px 0;border-radius:12px}a{color:#216c95}.poses{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:15px}.poses img{width:100%}</style>'


def state(report):
    if report['passed']:return '完整验收通过'
    if report['constructed']:
        reasons=[]
        if not report['compactness_passed']:reasons.append('整体尺寸超限')
        if any(not c['withdrawal']['clear'] for c in report['construction']['checks']):reasons.append('退出未通过')
        if any(not c['coupled_equilibrium_passed'] for c in report['construction']['checks']):reasons.append('承载未通过')
        return '实体已连接；'+'、'.join(reasons or ['几何验收未通过'])
    if report.get('preview_layout_passed'):return '头组摆放通过，实体构造未成功'
    return '有限搜索未找到通过的摆放；图中仅为候选'


def publish(group):
    output=group/'step4/reseated';path=output/'data/report.json';report=I.check_report(path)
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
    renderer=P.CPU.Renderer();rows=[];pictures=[];artifacts={}
    for k,task in enumerate(case.tasks):
        b,o=bases[k],offsets[k];translation=-b@o
        static=[]
        if body is not None:
            support=body.vertices@b.T+translation
            static.append(P.CPU.placed(P.piece(body,P.BLUE),b,translation))
        else:support=np.vstack([m.vertices for _,_,m in heads])@b.T+translation
        points=np.vstack([support,task.domain.mesh.vertices])
        floor=P.CPU.box(np.r_[np.ptp(points[:,:2],axis=0)+.04,.001],
            np.r_[(points[:,:2].min(0)+points[:,:2].max(0))/2,-.0006],'#f0f3f5');floor['unlit']=True
        parts=[P.CPU.placed(floor)]+static+[P.CPU.placed(P.piece(task.domain.mesh,P.GRAY))]
        parts += [P.CPU.placed(P.piece(m,P.ACTIVE if owner==k else P.IDLE,.000003),b,translation) for owner,_,m in heads]
        camera=P.CPU.fit(points,[-.8,-1.,.65],1.,padding=1.15)
        image=renderer.render(parts,camera,800,800)
        direction=-np.asarray(placement['directions'][k]);center=task.domain.mesh.vertices.mean(0)
        P.annotate_arrow(image,center,center+direction*.075,camera)
        page=Image.new('RGB',(800,945),'white');page.paste(image,(0,70));ink=ImageDraw.Draw(page)
        ink.text((20,12),f'{group.name} · {task.pose.replace("pose_","Pose ")}',font=P.font(27),fill='#294553')
        if body is not None:
            check=report['construction']['checks'][k]
            label=('退出通过' if check['withdrawal']['clear'] else '退出未通过')+' · '+('原载荷全部通过' if check['coupled_equilibrium_passed'] else '承载验收未通过')
            color='#2b775f' if check['withdrawal']['clear'] and check['coupled_equilibrium_passed'] else '#a8463b'
        else:
            label='仅接触组摆放候选 · 尚无连接实体';color='#a8463b'
        ink.text((20,874),label,font=P.font(22),fill=color)
        ink.text((20,910),'浅蓝：支撑　灰：物体　橙：本 pose 的头　紫：闲置头',font=P.font(19),fill='#496370')
        filename=f'placement_{task.pose}.png';page.save(output/filename)
        rows.append(dict(pose=task.pose,image=filename,constructed=body is not None,
            status=label,step3_covered=case.schedule['covered_counts'][k],
            fixture_rotation_world=b.tolist(),fixture_translation_world=translation.tolist(),
            object_exit_direction_world=direction.tolist()))
        pictures.append(page);artifacts['../'+filename]=I.sha256(output/filename)
    columns=min(3,len(pictures));height=100+945*math.ceil(len(pictures)/columns)
    canvas=Image.new('RGB',(800*columns,height),'white');ink=ImageDraw.Draw(canvas)
    ink.text((24,15),f'{group.name} · {state(report)}',font=P.font(30),fill='#294553' if report['passed'] else '#a8463b')
    subtitle='同一件支撑，各 pose 独立摆放；原接触组保持不变。' if body is not None else '各组原接触面的候选摆放；尚无支撑实体。'
    ink.text((24,59),subtitle,font=P.font(22),fill='#496370')
    for k,picture in enumerate(pictures):canvas.paste(picture,((k%columns)*800,100+(k//columns)*945))
    canvas.save(output/'placements.png');artifacts['../placements.png']=I.sha256(output/'placements.png')
    # Put the requested overview immediately inside this group's Step4 folder.
    shutil.copyfile(output/'placements.png',group/'step4/placements.png')
    artifacts['../../placements.png']=I.sha256(group/'step4/placements.png')
    links=''
    if body is not None:
        view=output/'data/viewer';view.mkdir(exist_ok=True)
        view_report=dict(report,diagnostic_only=False,physical_head_definition='independent_groups_reseated',
            presentation_description='同一件支撑，各 pose 采用各自的摆放；所有头和连接材料始终固定。')
        export_viewer(view,body,[(ident,m) for _,ident,m in heads],view_report,case.tasks,bases,offsets,{})
        html=(view/'index.html').read_text().replace('href="fixture.obj"','href="shape.obj"').replace('<a href="fixture_mm.stl" download>STL · mm</a>','')
        (output/'shape.html').write_text(html)
        artifacts['../shape.html']=I.sha256(output/'shape.html')
        links='<p><a href="shape.html">旋转查看各 pose 的摆放</a> · <a href="shape.obj">下载支撑 OBJ</a></p>'
    metric=f'体积 {report["volume_cm3"]:.1f} cm³；最大水平跨度 {report["metrics"]["maximum_horizontal_span_m"]*1000:.1f} mm。' if body is not None else '没有把未完成的头组候选当作连接支撑。'
    incomplete=[r for r in rows if r['step3_covered']<32768]
    note='<p>原 Step3 尚未覆盖全部载荷：'+ '，'.join(f'{r["pose"]}：{r["step3_covered"]}/32768' for r in incomplete)+'。本次保持这些头不变。</p>' if incomplete else ''
    cards=''.join(f'<section><h2>{r["pose"].replace("pose_","Pose ")}</h2><p>{r["status"]}</p><a href="{r["image"]}"><img src="{r["image"]}"></a></section>' for r in rows)
    html='<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'+STYLE
    html+=f'<h1>{group.name} · 各 pose 的摆放</h1><p>{state(report)}。{metric}</p>'+note+links+'<div class="poses">'+cards+'</div>'
    (output/'index.html').write_text(html);artifacts['../index.html']=I.sha256(output/'index.html')
    I.save(output/'data/visualization.json',dict(complete=True,poses=case.poses,per_pose=rows,
        image_size=[800,945],combined_size=list(canvas.size),videos_generated=False,
        provenance=dict(inputs=I.hashes([path]+([output/'shape.obj'] if body is not None else [])),
            code=I.hashes([Path(__file__),Path(P.CPU.__file__),Path(P.__file__)])),artifacts=artifacts))
    print('PLACEMENTS PUBLISHED',group.name,len(rows),'poses',flush=True)


def gallery(name):
    base=I.OUTPUTS/name;cards=[];rows=[]
    for group in groups(name):
        output=group/'step4/reseated';report=I.check_report(output/'data/report.json')
        visual=I.check_report(output/'data/visualization.json')
        relative=f'{group.name}/step4/reseated'
        cards.append(f'<section><h2><a href="{relative}/index.html">{group.name} · {state(report)}</a></h2><a href="{relative}/index.html"><img src="{relative}/placements.png"></a></section>')
        rows.append(dict(group=group.name,poses=report['poses'],constructed=report['constructed'],passed=report['passed'],
            status=report['status'],volume_cm3=report.get('volume_cm3'),images=len(visual['per_pose'])))
    (base/'step4_results.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'+STYLE+'<h1>B · 新 Step4：各 pose 的支撑摆放</h1><p>全部 8 组；同一件支撑，各 pose 独立就位与退出。浅蓝为支撑，灰色为物体，橙色为当前使用的头，紫色为闲置头。</p>'+''.join(cards))
    I.save(base/'reseated_results.json',dict(complete=True,groups=rows,videos_generated=False,
        provenance=dict(inputs=I.hashes([p/'step4/reseated/data/visualization.json' for p in groups(name)]),code=I.hashes([Path(__file__)])),
        artifacts={'step4_results.html':I.sha256(base/'step4_results.html')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('object',nargs='?',default='B');p.add_argument('--group')
    args=p.parse_args()
    for group in ([I.OUTPUTS/args.object/args.group] if args.group else groups(args.object)):publish(group)
    if not args.group:gallery(args.object)
