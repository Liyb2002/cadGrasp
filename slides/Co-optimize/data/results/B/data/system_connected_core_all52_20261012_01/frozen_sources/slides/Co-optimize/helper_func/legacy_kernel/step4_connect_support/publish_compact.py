"""Show the same compact fixture under each independently searched seating."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image,ImageDraw,ImageFont
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step4_connect_support import compact_layout as C,build_coupled_saddle as S
from step4_connect_support.video import mp4_writer
from step4_connect_support.codesign_port.fixture_view import export_viewer
from step4_connect_support.codesign_port import visual_details as V

spec=importlib.util.spec_from_file_location('compact_cpu',I.ROOT/'slides/reuse/code/render_cpu.py')
CPU=importlib.util.module_from_spec(spec);spec.loader.exec_module(CPU)
BLUE,GRAY,ACTIVE,IDLE='#8fc9e8','#aeb8bf','#e6a13f','#9c76bd'


def font(size):return ImageFont.truetype('/System/Library/Fonts/STHeiti Medium.ttc',size)
def piece(mesh,color,bias=0):return CPU.piece(dict(v=mesh.vertices.ravel(),f=mesh.faces.ravel()),color,bias=bias)


def annotate_arrow(picture,start,end,camera,label=True):
    focus,basis,span=camera;size=picture.height
    p=(np.array([start,end])-focus)@basis.T
    xy=np.c_[picture.width/2+p[:,0]*size/span,size/2-p[:,1]*size/span]
    ink=ImageDraw.Draw(picture);V.arrow(ink,*xy,color='#7b54a5',width=4)
    if label:ink.text(tuple(xy.mean(0)-[0,24]),'物体退出',anchor='mm',font=font(20),fill='#7b54a5')


def publish(output):
    output=Path(output).resolve();report_path=output/'data/report.json'
    report=I.check_report(report_path)
    if not report['constructed']:raise ValueError('No compact support to display')
    case=C.load_case(output)
    body=trimesh.load(output/'shape.obj',force='mesh',process=False)
    bases=np.asarray(report['placement']['bases']);offsets=np.asarray(report['placement']['offsets'])
    directions=np.asarray(report['placement']['directions'])
    heads=[]
    for owner,group in enumerate(case.groups):
        for contact in group:
            points=contact['triangles_m'].reshape(-1,3)@bases[owner]+offsets[owner]
            mesh=trimesh.Trimesh(points,np.arange(len(points)).reshape(-1,3),process=False)
            heads.append((owner,contact['candidate_id'],mesh))
    view_report=dict(report,source_schedule=str((case.source/'schedule.json').relative_to(I.ROOT)),
        diagnostic_only=False,presentation_description='同一件紧凑支撑，不同 pose 独立就位；原接触组保持不变。',
        physical_head_definition='unshared_heads_compact_independent_seating')
    view=output/'data/viewer';view.mkdir(exist_ok=True)
    colors={ident:ACTIVE if owner==0 else IDLE for owner,ident,_ in heads}
    export_viewer(view,body,[(ident,m) for _,ident,m in heads],view_report,case.tasks,bases,offsets,colors)
    html=(view/'index.html').read_text().replace('href="fixture.obj"','href="shape.obj"').replace('<a href="fixture_mm.stl" download>STL · mm</a>','')
    (output/'shape.html').write_text(html)
    renderer=CPU.Renderer();support_piece=piece(body,BLUE)
    pictures=[];rows=[];artifacts={}
    for k,task in enumerate(case.tasks):
        b,o=bases[k],offsets[k];rotation=b;translation=-b@o
        fixture=body.vertices@b.T+translation
        direction=-directions[k];center=task.domain.mesh.vertices.mean(0)
        static=[CPU.placed(support_piece,rotation,translation)]
        patches=[CPU.placed(piece(m,ACTIVE if owner==k else IDLE,.000003),rotation,translation) for owner,_,m in heads]
        object_piece=piece(task.domain.mesh,GRAY)
        # Actual task-world object is used here. Each task has its own object
        # pose relative to the same fixture; never substitute a common copy.
        points=np.vstack([fixture,task.domain.mesh.vertices])
        floor=CPU.box(np.r_[np.ptp(points[:,:2],axis=0)+.04,.001],np.r_[(points[:,:2].min(0)+points[:,:2].max(0))/2,-.0006],'#f0f3f5');floor['unlit']=True
        static=[CPU.placed(floor)]+static
        camera=CPU.fit(points,[-.8,-1.,.65],1.,padding=1.17)
        image=renderer.render(static+[CPU.placed(object_piece)]+patches,camera,800,800)
        annotate_arrow(image,center,center+direction*.09,camera)
        page=Image.new('RGB',(800,910),'white');page.paste(image,(0,65))
        ink=ImageDraw.Draw(page)
        ink.text((22,15),f'{case.pair.name} · {task.pose.replace("pose_","Pose ")}',font=font(27),fill='#304a57')
        check=report['construction']['checks'][k]
        state=('退出通过' if check['withdrawal']['clear'] else '退出未通过')+' · '+('原载荷全通过' if check['coupled_equilibrium_passed'] else '承载验收未通过')
        ink.text((22,862),state,font=font(22),fill='#2b775f' if check['coupled_equilibrium_passed'] else '#a8463b')
        filename=f'shape_{task.pose}.png';page.save(output/filename);pictures.append(page);artifacts[filename]=I.sha256(output/filename)
        # Full continuous sweep is independently certified in construction.
        # Instantaneous checks below are additional replay checks, not its basis.
        solid=S.solid(trimesh.Trimesh(fixture,body.faces,process=False))
        obj=S.solid(task.domain.mesh);checks=[]
        for mm in [0,.5,1,2,5,10,20,50,100,200,500]:
            overlap=abs(float((solid^obj.translate((direction*mm/1000/S.SCALE).tolist())).volume()))*S.SCALE**3
            checks.append(dict(distance_mm=mm,overlap_m3=overlap))
        if check['withdrawal']['clear']:assert max(r['overlap_m3'] for r in checks)<C.TOL
        side=np.r_[-direction[1],direction[0],.6]
        full_camera=CPU.fit(np.vstack([points,task.domain.mesh.vertices+direction*.5]),side,1.,padding=1.14)
        close_camera=CPU.fit(np.vstack([points,task.domain.mesh.vertices+direction*.02]),side,1.,padding=1.16)
        size=700;video=output/f'exit_{task.pose}.mp4'
        amounts=np.r_[np.zeros(12),np.linspace(0,20,32),np.linspace(32,500,40),np.full(12,500.)]
        with mp4_writer(video,fps=12) as writer:
            for mm in amounts:
                left=renderer.render(static+[CPU.placed(object_piece,point=direction*mm/1000)]+patches,full_camera,size,size)
                annotate_arrow(left,center,center+direction*.5,full_camera)
                near=min(mm,20.)
                right=renderer.render(static+[CPU.placed(object_piece,point=direction*near/1000)]+patches,close_camera,size,size)
                annotate_arrow(right,center,center+direction*.09,close_camera)
                frame=Image.new('RGB',(1400,830),'white');frame.paste(left,(0,90));frame.paste(right,(700,90))
                ink=ImageDraw.Draw(frame)
                ink.text((20,14),f'{case.pair.name} · {task.pose.replace("pose_","Pose ")} · 同一支撑，独立就位',font=font(28),fill='#294553')
                ink.text((20,54),f'全程退出 · {mm:g} mm',font=font(22),fill='#294553')
                ink.text((720,54),f'起始退出近景 · {near:g} mm'+('（定格）' if mm>20 else ''),font=font(22),fill='#294553')
                ink.text((20,792),'浅蓝：支撑　灰：物体　橙：本 pose 的头　紫：其他 pose 的头',font=font(20),fill='#496370')
                ink.text((960,792),state,font=font(20),fill='#2b775f' if check['coupled_equilibrium_passed'] else '#a8463b')
                writer.append_data(np.asarray(frame))
        artifacts[video.name]=I.sha256(video)
        rows.append(dict(pose=task.pose,video=video.name,image=filename,frames=len(amounts),fps=12,
            withdrawal_passed=check['withdrawal']['clear'],loads_passed=check['coupled_equilibrium_passed'],
            instantaneous_replay=checks,object_direction_world=direction.tolist(),
            support_stationary=True,near_view_freezes_at_mm=20.))
        print('COMPACT VIDEO',case.pair.name,task.pose,state,flush=True)
    camera=CPU.fit(body.vertices,[-.8,-1.,.65],1.,padding=1.17)
    image=renderer.render([CPU.placed(support_piece)]+[CPU.placed(piece(m,colors[ident],.000003)) for _,ident,m in heads],camera,800,800)
    panel=Image.new('RGB',(800,910),'white');panel.paste(image,(0,65))
    ImageDraw.Draw(panel).text((22,15),'同一件支撑',font=font(27),fill='#304a57');pictures.append(panel)
    combined=Image.new('RGB',(2400,910),'white')
    for i,picture in enumerate(pictures):combined.paste(picture,(800*i,0))
    combined.save(output/'shape.png');artifacts['shape.png']=I.sha256(output/'shape.png')
    label='完整验收通过' if report['passed'] else '实体已连接；完整验收未通过'
    cards=''.join(f'<section id="{r["pose"]}"><h2>{r["pose"].replace("pose_","Pose ")}</h2><p>退出：{"通过" if r["withdrawal_passed"] else "未通过"}；原载荷：{"全部通过" if r["loads_passed"] else "未通过"}</p><video controls preload="metadata" src="{r["video"]}" poster="{r["image"]}"></video><p><a href="{r["video"]}">下载退出视频</a></p></section>' for r in rows)
    style='<style>body{font:16px system-ui;max-width:1500px;margin:28px auto;padding:0 20px;color:#294553;background:#f5f8fa}p{line-height:1.6}section{background:white;padding:20px;margin:20px 0;border-radius:12px}img,video{width:100%}a{color:#216c95}</style>'
    introduction=f'<h1>{case.pair.name} · 紧凑独立就位</h1><p>{label}；体积 {report["volume_cm3"]:.1f} cm³；最大水平跨度 {report["metrics"]["maximum_horizontal_span_m"]*1000:.1f} mm。</p><p>各组头随整个支撑固定，各 pose 的物体相对支撑摆放不同。橙/紫区分图中正在使用的头与其他头；物体退出时，支撑固定。每个 pose 使用原始 32,768 个载荷验收。</p><p><a href="shape.html">旋转查看支撑及各 pose</a> · <a href="shape.obj">OBJ</a></p><img src="shape.png">'
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'+style+introduction+cards)
    for filename in ['shape.html','index.html']:artifacts[filename]=I.sha256(output/filename)
    I.save(output/'data/visualization.json',dict(complete=True,poses=case.poses,per_pose=rows,
        physical_geometry_unchanged=True,provenance=dict(inputs=I.hashes([report_path,output/'shape.obj']),
        code=I.hashes([Path(__file__),Path(CPU.__file__),Path(export_viewer.__code__.co_filename)])),
        artifacts={'../'+n:h for n,h in artifacts.items()}))


def gallery(name):
    base=I.OUTPUTS/name;cards=[]
    for group in ['pose3+6','pose5+7']:
        output=base/group/'step4/compact';r=I.check_report(output/'data/report.json')
        if r['constructed']:
            I.check_report(output/'data/visualization.json')
            state='完整验收通过' if r['passed'] else '退出通过；承载验收未通过' if all(c['withdrawal']['clear'] for c in r['construction']['checks']) else '未通过'
            cards.append(f'<section><h2><a href="{group}/step4/compact/index.html">{group} · {state}</a></h2><p>{r["volume_cm3"]:.1f} cm³；最大水平跨度 {r["metrics"]["maximum_horizontal_span_m"]*1000:.1f} mm</p><a href="{group}/step4/compact/index.html"><img src="{group}/step4/compact/shape.png"></a></section>')
        else:cards.append(f'<section><h2>{group}</h2><p>有限搜索未生成实体</p></section>')
    style='<style>body{font:17px system-ui;max-width:1500px;margin:30px auto;padding:0 20px;color:#294553}img{width:100%}section{padding:18px;background:#f5f8fa;margin:20px 0}a{color:#216c95}</style>'
    (base/'compact_layout.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'+style+'<h1>B：紧凑支撑、各 pose 独立就位</h1><p>同一件支撑在不同摆放下接触不同的头组。保持原 Step3 接触和载荷，联合搜索整组头的摆放与退出方向。</p>'+''.join(cards))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('outputs',type=Path,nargs='*');p.add_argument('--gallery')
    args=p.parse_args()
    for output in args.outputs:publish(output)
    if args.gallery:gallery(args.gallery)
