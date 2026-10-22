"""Explain each saved task's actual ground feet and shared-face demands.

Only diagnostic geometry is drawn. The accepted fixtures, poses, certificates,
and published result/video are unchanged. Pressure polygons and contact hulls
are virtual planar envelopes; only saved coplanar triangles are real contacts.
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import ConvexHull
import trimesh

from whole_task.step1.common import ROOT, DEFAULT_OUTPUT, sha256, write_json
from whole_task.step1.needle_render import sector_lines
from whole_task.operations.geometry import moved_mesh
from whole_task.operations.state import transform
from whole_task.step3 import independent_geometry as G
from whole_task.step3.run_membranes import load_model
from whole_task.step3.render_independent import Renderer, head_context, BLUE
from whole_task.step4.ground_oracle import pressure_rows, ground_enclosure
from whole_task.step4.membrane_search import read_group
from .video import raster
from .height_reference import model_for_report

GREEN=(.12,.59,.36)
LEG_GRAY=(.76,.79,.81)
INK='#29475c'
PINK='#d94c89'
PURPLE='#8574b5'
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if not Path(FONT).exists():FONT='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'


def font(size):return ImageFont.truetype(FONT,size)


def same_floor_indices(material,index,tolerance=1e-8):
    """Group a physical material face; XY translation/yaw is allowed to differ."""
    normals=material[:,:3,2]
    offsets=np.einsum('ij,ij->i',normals,material[:,:3,3])
    return [j for j in range(len(material)) if np.linalg.norm(normals[j]-normals[index])<tolerance
        and abs(offsets[j]-offsets[index])<tolerance]


def ground_triangles(mesh,T):
    world=transform(mesh.vertices,T);tri=world[mesh.faces]
    tri=tri[np.abs(tri[:,:,2]).max(1)<=1e-8]
    a=tri[:,1,:2]-tri[:,0,:2];b=tri[:,2,:2]-tri[:,0,:2]
    return tri[np.abs(a[:,0]*b[:,1]-a[:,1]*b[:,0])*.5>1e-15]


def polygon(points):return points[ConvexHull(points).vertices]


def dimensions(points):return np.ptp(points,axis=0)*1000.


def dim_text(points):
    a,b=dimensions(points)
    return f'{a:.0f} × {b:.0f} mm'


def plot_ground(record,span,size=(600,550)):
    fig=Figure(figsize=(size[0]/100,size[1]/100),dpi=100,facecolor='white')
    canvas=FigureCanvasAgg(fig);ax=fig.add_axes([.14,.14,.83,.80])
    current=record['required'];actual=record['contact_hull'];bounds=np.array([actual.min(0),actual.max(0)])
    center=bounds.mean(0)*1000
    ax.set_xlim(center[0]-span/2,center[0]+span/2);ax.set_ylim(center[1]-span/2,center[1]+span/2)
    ax.set_aspect('equal');ax.xaxis.set_major_locator(MultipleLocator(100));ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.grid(color='#e7ebee',linewidth=.7);ax.set_axisbelow(True)
    for other in record['other_demands']:
        p=other['points']*1000
        ax.fill(p[:,0],p[:,1],facecolor=PURPLE,edgecolor=PURPLE,alpha=.12,linewidth=1.)
        closed=np.vstack([p,p[0]])
        ax.plot(closed[:,0],closed[:,1],color=PURPLE,linewidth=1.1,alpha=.85)
        c=p.mean(0);ax.text(c[0],c[1],other['id'],fontsize=10,color='#78639f',ha='center',va='center')
    if record['other_demands']:
        p=record['shared_hull']*1000;closed=np.vstack([p,p[0]])
        ax.plot(closed[:,0],closed[:,1],color=PURPLE,linewidth=1.3,linestyle=':')
    p=current*1000
    ax.fill(p[:,0],p[:,1],facecolor=PINK,edgecolor=PINK,alpha=.25,linewidth=1.4)
    closed=np.vstack([p,p[0]])
    ax.plot(closed[:,0],closed[:,1],color=PINK,linewidth=1.8)
    ax.add_collection(PolyCollection(record['ground_triangles'][:,:,:2]*1000,
        facecolor='#239963',edgecolor='#239963',linewidth=.35,zorder=4))
    closed=np.vstack([actual,actual[0]])*1000
    ax.plot(closed[:,0],closed[:,1],color='#168653',linewidth=1.5,linestyle='--',zorder=3)
    ax.set_xlabel('x (mm)',fontsize=11,color='#5e7381');ax.set_ylabel('y (mm)',fontsize=11,color='#5e7381')
    ax.tick_params(labelsize=10,colors='#69808c',length=3)
    for spine in ax.spines.values():spine.set_color('#cdd7dd')
    canvas.draw()
    return Image.fromarray(np.asarray(canvas.buffer_rgba())[:,:,:3].copy())


def collect(folder):
    report_path=folder/'report.json';report=json.loads(report_path.read_text())
    model,_,_=load_model(ROOT/'whole_task/data'/report['object']/'step3_membranes')
    model=model_for_report(model,report)
    records=[];families=[]
    for family,group in enumerate(report['groups']):
        q,mesh=read_group(group);_,core=read_group(group['core'])
        part=model.subset(group['members']);rows=pressure_rows(part,q)
        checks=ground_enclosure(mesh,rows)
        if not all(row['passed'] for row in checks):raise ValueError('Saved feet no longer enclose actual pressures')
        solid=G.solid(mesh);core_solid=G.solid(core)
        retained=solid^core_solid;added=solid-core_solid
        if abs(G.volume(retained)+G.volume(added)-mesh.volume)>1e-12:
            raise ValueError('Diagnostic core/foot partition changes total material')
        groups=[];assigned=set()
        for i in range(len(q.poses)):
            if i not in assigned:
                ids=same_floor_indices(q.material,i);assigned.update(ids);groups.append(ids)
        families.append(dict(set=family+1,final_volume_cm3=mesh.volume*1e6,
            retained_step4_membrane_cm3=G.volume(retained)*1e6,
            added_material_cm3=G.volume(added)*1e6,ground_face_count=len(groups)))
        for local,owner in enumerate(group['members']):
            T=np.linalg.inv(q.material[local]);needed=rows[local]['hull']
            ids=same_floor_indices(q.material,local);other=[];demands=[]
            for j in ids:
                p=transform(rows[j]['ground_material'],T)
                if np.abs(p[:,2]).max()>2e-8:raise ValueError('Shared-face demand is not on this actual ground plane')
                demands.append(p[:,:2])
                if j!=local:other.append(dict(id=rows[j]['id'],points=p[:,:2]))
            shared=polygon(np.vstack(demands));contact=ground_triangles(mesh,T)
            hull=polygon(contact[:,:,:2].reshape(-1,2))
            equations=ConvexHull(hull).equations
            gap=float((shared@equations[:,:2].T+equations[:,2]).max())
            if gap>1e-8:raise ValueError('Current ground hull does not cover the transformed shared-face needs')
            # Green is exactly material in the bottom 5 mm, rather than an
            # invented owner assignment for connectors used by several poses.
            low=solid.trim_by_plane(tuple(-T[2,:3]),float((T[2,3]-.005)/G.SCALE))
            # A strand passing close to the ground is not a grounded foot.
            # Retain the bottom-band components with real coplanar area only.
            for piece in low.decompose():
                if float(piece.volume())*G.SCALE**3>1e-15 and not len(ground_triangles(G.mesh(piece),T)):
                    low=low-piece
            upper=solid-low
            skin=upper^core_solid;legs=upper-core_solid
            pieces=[]
            for value,color in [(skin,BLUE),(legs,LEG_GRAY),(low,GREEN)]:
                if not value.is_empty():pieces.append((moved_mesh(G.mesh(value),T),color,None))
            body=moved_mesh(model.source,q.poses[local])
            face=next(i for i,items in enumerate(groups) if local in items)
            records.append(dict(id=model.ids[owner],owner=owner,set=family+1,face=chr(65+face),
                ground_triangles=contact,required=needed,other_demands=other,shared_hull=shared,
                contact_hull=hull,pieces=pieces,body=body,full_world=moved_mesh(mesh,T),
                same_face_tasks=[rows[j]['id'] for j in ids],minimum_body_height_mm=float(body.bounds[0,2]*1000),
                pressure_area_cm2=float(ConvexHull(needed).volume*1e4),
                shared_pressure_area_cm2=float(ConvexHull(shared).volume*1e4),
                actual_contact_hull_area_cm2=float(ConvexHull(hull).volume*1e4),
                actual_ground_area_cm2=float(trimesh.triangles.area(contact).sum()*1e4),
                source_mesh_sha256=group['sha256'],ground_check=checks[local]))
    records.sort(key=lambda row:row['owner'])
    return model,records,families,report_path


def render(folder,output):
    model,records,families,report_path=collect(folder)
    cardw,cardh,columns=1200,850,3;header=265;footer=90
    rows=math.ceil(len(records)/columns)
    canvas=Image.new('RGB',(cardw*columns,header+cardh*rows+footer),'white');draw=ImageDraw.Draw(canvas)
    draw.text((35,20),'Step5｜每个 task 的地脚与压力需求',font=font(48),fill=INK)
    head_height=float(model.head[2]*1000)
    draw.text((35,90),f'针根离地 {head_height:.0f} mm；同一组始终使用同一个完整实体。每张图显示当前姿态真正落地的部分。',font=font(29),fill='#5b7484')
    draw.text((35,145),'3D：蓝＝保留的薄壳　灰＝其余连杆　绿＝当前落地端（接地脚底部 5 mm 着色）',font=font(28),fill='#526b7b')
    draw.text((35,194),'俯视：粉＝当前 task 压力凸包　紫＝同接地面其他 task 的需求　绿块＝实际接触　绿虚线＝触地点凸包',font=font(27),fill='#526b7b')
    span=max(float(np.ptp(row['contact_hull'],axis=0).max()*1000) for row in records)*1.17
    renderer=Renderer(600,550);contexts=head_context(model.workspace);sector=sector_lines(model.workspace)
    try:
        for index,record in enumerate(records):
            row,column=divmod(index,columns)
            if row==rows-1 and len(records)%columns==1:column=1
            x,y=column*cardw,header+row*cardh
            draw.text((x+24,y+4),f"{record['id']}  ·  支撑 {record['set']}  ·  接地面 {record['face']}",font=font(34),fill=INK)
            draw.text((x+24,y+58),'当前哪些脚落地',font=font(23),fill='#6b818e')
            draw.text((x+645,y+58),'地面俯视（所有图同比例）',font=font(23),fill='#6b818e')
            body=record['body'];mask=np.zeros(len(model.source.faces),bool)
            mask[model.records[record['owner']]['working_face_ids']]=True
            ground=trimesh.creation.box([1.8,1.8,.004]);ground.apply_translation([0,-.4,-.00202])
            fit=np.vstack([record['full_world'].vertices,body.vertices,sector,*[m.vertices for m,_,_ in contexts]])
            scene=renderer.draw([(ground,(.96,.97,.97),None),*record['pieces'],
                (body,raster.GRAY,mask),*contexts],fit,[.65,1.4,1.15],sector)
            canvas.paste(scene,(x,y+90));canvas.paste(plot_ground(record,span),(x+600,y+90))
            draw.text((x+30,y+648),f"本 task 的压力区：{dim_text(record['required'])}",font=font(27),fill=PINK)
            draw.text((x+30,y+692),f"同面合并需求：{dim_text(record['shared_hull'])}　({' / '.join(record['same_face_tasks'])})",font=font(25),fill=PURPLE)
            draw.text((x+30,y+736),f"实际落地凸包：{dim_text(record['contact_hull'])}",font=font(27),fill='#168653')
            draw.text((x+635,y+736),f"物体最低点离地 {record['minimum_body_height_mm']:.0f} mm",font=font(24),fill='#6b818e')
            draw.line([(x+20,y+820),(x+cardw-20,y+820)],fill='#e4e9ed',width=2)
            print('TASK FEET',record['id'],'pressure',dim_text(record['required']),
                'shared',dim_text(record['shared_hull']),'actual',dim_text(record['contact_hull']),flush=True)
    finally:renderer.close()
    # Explain the particularly large second fixture in the two otherwise
    # empty cards. Its common-floor needs really span several separate seats.
    last=header+(rows-1)*cardh
    example=max(records,key=lambda row:row['shared_pressure_area_cm2'])
    minimum=min(row['minimum_body_height_mm'] for row in records)
    maximum=max(row['minimum_body_height_mm'] for row in records)
    for offset,text,size,color in [(40,'为什么脚又高又宽？',34,INK),
        (115,f'高：针根固定在离地 {head_height:.0f} mm。',27,'#526b7b'),
        (165,f'物体最低点仍离地 {minimum:.0f}–{maximum:.0f} mm，',27,'#526b7b'),
        (210,'需要连杆把薄壳接到地面。',27,'#526b7b'),
        (290,'宽：同一落地面服务多个不同座位。',27,'#526b7b'),
        (345,f"例如 {' / '.join(example['same_face_tasks'])}，",27,'#526b7b'),
        (395,f"合并压力区约 {dim_text(example['shared_hull'])}，",27,PURPLE),
        (445,f"当前触地点凸包约 {dim_text(example['contact_hull'])}。",27,'#168653'),
        (530,'紫色区域换到支撑坐标后会分散，',27,'#526b7b'),
        (575,'不是每个 task 单独都要这么大。',27,'#526b7b')]:
        draw.text((35,last+offset),text,font=font(size),fill=color)
    draw.text((2*cardw+35,last+40),'当前实体的材料构成',font=font(34),fill=INK)
    for i,family in enumerate(families):
        yy=last+120+i*180
        draw.text((2*cardw+35,yy),f"支撑 {family['set']}：共 {family['final_volume_cm3']:.1f} cm³",font=font(29),fill=INK)
        draw.text((2*cardw+35,yy+53),f"保留薄壳 {family['retained_step4_membrane_cm3']:.1f} cm³",font=font(27),fill='#337eae')
        draw.text((2*cardw+35,yy+98),f"新增连杆 / 地脚 {family['added_material_cm3']:.1f} cm³",font=font(27),fill='#73848e')
    draw.text((2*cardw+35,last+530),'虚线凸包是虚拟范围，内部无需填底板。',font=font(26),fill='#526b7b')
    draw.text((2*cardw+35,last+580),'这里只解释现有解，不声称地脚已最小。',font=font(26),fill='#526b7b')
    draw.text((35,canvas.height-65),'绿色脚端在其他姿态可能离地；灰色连杆也可能服务其他 task。原有实体、工作 pose 和受力证书保持原样。',font=font(27),fill='#6b818e')
    output.parent.mkdir(parents=True,exist_ok=True);canvas.save(output)
    data=[]
    for record in records:
        keys=['id','set','face','same_face_tasks','minimum_body_height_mm','pressure_area_cm2',
            'shared_pressure_area_cm2','actual_contact_hull_area_cm2','actual_ground_area_cm2',
            'source_mesh_sha256','ground_check']
        row={key:record[key] for key in keys}
        for key in ['required','shared_hull','contact_hull']:row[key+'_xy_m']=record[key].tolist()
        for key in ['required','shared_hull','contact_hull']:row[key+'_bounds_mm']=dimensions(record[key]).tolist()
        row['other_same_face_demands']=[dict(id=d['id'],xy_m=d['points'].tolist()) for d in record['other_demands']]
        data.append(row)
    result=dict(status='passed',task_count=len(records),same_scale_ground_span_mm=span,
        needle_root_height_mm=head_height,green_display_band_mm=5.,
        actual_contacts_from_nonzero_area_coplanar_triangles=True,shared_face_demands_transformed_through_material=True,
        minimum_foot_design_claimed=False,existing_geometry_modified=False,families=families,tasks=data,
        figure=str(output.resolve()),figure_sha256=sha256(output),
        input_sha256={str(report_path):sha256(report_path),str(Path(__file__).resolve()):sha256(__file__)})
    write_json(folder/'task_feet_diagnostic.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--folder',type=Path)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT/'B/step5/feet_by_task.png')
    args=parser.parse_args()
    if args.folder is None:
        args.folder=Path(json.loads((ROOT/'whole_task/data/B/step5_active.json').read_text())['workdir'])
    render(args.folder,args.output)
