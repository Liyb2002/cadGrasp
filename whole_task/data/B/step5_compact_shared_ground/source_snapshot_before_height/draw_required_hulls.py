"""Draw required pressure hulls per physical fixture and installation plane."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial import ConvexHull

from whole_task.step1.common import ROOT,DEFAULT_OUTPUT,sha256,write_json
from whole_task.operations.state import transform
from whole_task.step3.run_membranes import load_model
from whole_task.step4.ground_oracle import pressure_rows
from whole_task.step4.membrane_search import read_group
from .draw_task_feet import same_floor_indices

FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
COLORS=['#d46a44','#5178b9','#439683','#9766af','#b6932e','#c25c87']


def collect(folder):
    path=folder/'report.json';report=json.loads(path.read_text())
    model,_,_=load_model(ROOT/'whole_task/data'/report['object']/'step3_membranes')
    families=[]
    for number,g in enumerate(report['groups'],1):
        q,_=read_group(g);cq,core=read_group(g['core'])
        for key,value in q.arrays().items():np.testing.assert_array_equal(value,cq.arrays()[key])
        rows=pressure_rows(model.subset(g['members']),q)
        reference=np.linalg.inv(q.material[0]);faces=[];assigned=set()
        for i,row in enumerate(rows):
            if i in assigned:continue
            ids=same_floor_indices(q.material,i);assigned.update(ids)
            T=np.linalg.inv(q.material[i]);polygons=[];tasks=[]
            for j in ids:
                # Demands remain in their real material locations. Sharing
                # a plane does not mean sharing XY coordinates or full poses.
                material=rows[j]['ground_material'];local=transform(material,T)
                if np.abs(local[:,2]).max()>2e-8:raise ValueError('Required hull leaves its ground plane')
                polygons.append(local[:,:2]);tasks.append(dict(id=rows[j]['id'],
                    material_xyz_m=material,plane_xy_m=local[:,:2]))
                np.testing.assert_allclose(transform(material,np.linalg.inv(q.material[j]))[:,:2],
                    rows[j]['hull'],atol=1e-12)
            all_points=np.vstack(polygons);h=ConvexHull(all_points);hull=all_points[h.vertices]
            if (all_points@h.equations[:,:2].T+h.equations[:,2]).max()>1e-10:
                raise ValueError('Merged hull loses an original pressure demand')
            material=transform(np.c_[hull,np.zeros(len(hull))],q.material[i])
            faces.append(dict(label=chr(65+len(faces)),tasks=tasks,plane_xy_m=hull,
                material_xyz_m=material,view_xyz_m=transform(material,reference),
                width_height_mm=np.ptp(hull,axis=0)*1000.,area_cm2=float(h.volume*1e4),
                color=COLORS[len(faces)%len(COLORS)]))
        if assigned!=set(range(len(rows))):raise ValueError('Missing task pressure hull')
        families.append(dict(number=number,members=g['members'],reference=reference,
            core=core,core_vertices=transform(core.vertices,reference),faces=faces))
    return families,path


def render(folder,output):
    font_manager.fontManager.addfont(FONT)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=FONT).get_name(),
        'axes.unicode_minus':False,'font.size':10})
    families,report_path=collect(folder)
    if len(families)!=2:raise ValueError('This overview is arranged for the current two fixtures')
    fig=plt.figure(figsize=(20,13.5),dpi=180,facecolor='white')
    fig.suptitle('Step5｜每个支撑需要覆盖的接地凸包',fontsize=23,color='#29475c',y=.979)
    fig.text(.5,.938,'上：各落地面的凸包放回同一支撑坐标系　　下：各面平面展开（统一比例，mm）',
        ha='center',fontsize=12,color='#617684')
    clouds=[np.vstack([f['core_vertices'],*[p['view_xyz_m'] for p in f['faces']]])*1000. for f in families]
    span=max(np.ptp(points,axis=0).max() for points in clouds)*1.12
    planar_span=max(p['width_height_mm'].max() for f in families for p in f['faces'])*1.15
    for col,(family,cloud) in enumerate(zip(families,clouds)):
        left=col*.5
        ax=fig.add_axes([left+.025,.535,.45,.365],projection='3d',proj_type='ortho')
        vertices=family['core_vertices']*1000.
        skin=Poly3DCollection(vertices[family['core'].faces],facecolor='#a9bbc7',
            edgecolor='none',alpha=.42,rasterized=True)
        ax.add_collection3d(skin)
        for face in family['faces']:
            p=face['view_xyz_m']*1000.;color=face['color']
            ax.add_collection3d(Poly3DCollection([p],facecolor=color,edgecolor=color,
                linewidth=1.5,alpha=.23))
            closed=np.vstack([p,p[0]])
            ax.plot(*closed.T,color=color,lw=1.8)
            for task in face['tasks']:
                t=transform(task['material_xyz_m'],family['reference'])*1000.;t=np.vstack([t,t[0]])
                ax.plot(*t.T,color=color,lw=.65,alpha=.7)
            center=p.mean(0)
            ax.text(*center,face['label'],color=color,fontsize=13,fontweight='bold',
                bbox=dict(facecolor='white',edgecolor='none',alpha=.75,pad=1.))
        center=(cloud.min(0)+cloud.max(0))/2
        ax.set_xlim(center[0]-span/2,center[0]+span/2)
        ax.set_ylim(center[1]-span/2,center[1]+span/2)
        ax.set_zlim(center[2]-span/2,center[2]+span/2)
        ax.set_box_aspect([1,1,1]);ax.view_init(elev=55,azim=-100)
        ax.set_xlabel('X (mm)',fontsize=8,labelpad=0);ax.set_ylabel('Y (mm)',fontsize=8,labelpad=0)
        ax.set_zlabel('Z (mm)',fontsize=8,labelpad=0);ax.tick_params(labelsize=7,pad=0)
        for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
            axis.set_major_locator(plt.MaxNLocator(4));axis.pane.fill=False
            axis._axinfo['grid'].update(color='#dfe5e9',linewidth=.5)
        ax.set_title(f"支撑 {family['number']} · {len(family['members'])} 个 task · {len(family['faces'])} 个落地面",
            fontsize=16,color='#29475c',pad=8)
        for k,face in enumerate(family['faces']):
            row,column=divmod(k,3)
            panel=fig.add_axes([left+.033+column*.155,.29-row*.225,.132,.175])
            p=face['plane_xy_m']*1000.;color=face['color'];closed=np.vstack([p,p[0]])
            panel.fill(p[:,0],p[:,1],color=color,alpha=.19)
            panel.plot(*closed.T,color=color,lw=1.8)
            for task in face['tasks']:
                t=task['plane_xy_m']*1000.;closed=np.vstack([t,t[0]])
                panel.plot(*closed.T,color=color,lw=.8,alpha=.8)
                panel.text(*t.mean(0),task['id'],ha='center',va='center',fontsize=7,
                    color='#3c4d58',bbox=dict(facecolor='white',edgecolor='none',alpha=.65,pad=.6))
            center=(p.min(0)+p.max(0))/2
            panel.set_xlim(center[0]-planar_span/2,center[0]+planar_span/2)
            panel.set_ylim(center[1]-planar_span/2,center[1]+planar_span/2)
            panel.set_aspect('equal');panel.grid(color='#e3e8eb',lw=.5);panel.set_axisbelow(True)
            panel.tick_params(labelsize=7,colors='#72838c',length=2)
            for spine in panel.spines.values():spine.set_color('#d4dde2')
            names=', '.join(sorted(t['id'] for t in face['tasks']));w,h=face['width_height_mm']
            panel.set_title(f"{face['label']}｜{names}\n{w:.0f} × {h:.0f} mm",fontsize=8.5,color=color,pad=6)
    fig.legend(handles=[Patch(facecolor='#a9bbc7',alpha=.5,label='薄壳（定位参考）'),
        Line2D([],[],color='#7d8990',lw=.8,label='细线：单个 task 的需求'),
        Patch(facecolor=COLORS[0],edgecolor=COLORS[0],alpha=.3,label='彩色面／粗线：同一落地面的合并凸包')],
        loc='center',bbox_to_anchor=(.5,.495),ncol=3,frameon=False,fontsize=11)
    fig.text(.5,.019,'各落地面分别求凸包；凸包表示需要覆盖的承压范围，面内不要求填满材料。',
        ha='center',fontsize=11,color='#617684')
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,facecolor='white');plt.close(fig)
    metadata=[]
    for family in families:
        faces=[]
        for face in family['faces']:
            faces.append(dict(label=face['label'],task_ids=[t['id'] for t in face['tasks']],
                required_hull_support_xyz_m=face['material_xyz_m'].tolist(),
                required_hull_plane_xy_m=face['plane_xy_m'].tolist(),
                width_height_mm=face['width_height_mm'].tolist(),area_cm2=face['area_cm2'],
                task_demands=[dict(id=t['id'],support_xyz_m=t['material_xyz_m'].tolist()) for t in face['tasks']]))
        metadata.append(dict(set=family['number'],members=family['members'],ground_faces=faces))
    result=dict(status='passed',task_count=sum(len(f['members']) for f in families),
        group_count=len(families),groups=metadata,all_task_pressure_demands_preserved=True,
        existing_geometry_modified=False,input_report_sha256=sha256(report_path),
        source_sha256=sha256(__file__),figure=str(output.resolve()),figure_sha256=sha256(output))
    write_json(folder/'required_hulls.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--folder',type=Path)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT/'B/step5/required_hulls.png')
    args=parser.parse_args()
    folder=args.folder or Path(json.loads((ROOT/'whole_task/data/B/step5_active.json').read_text())['workdir'])
    render(folder,args.output)
