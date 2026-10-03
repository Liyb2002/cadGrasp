"""One sheet: actual historical Pose 3 support, head loads and load comparison."""
from pathlib import Path
import json
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
GROUP=ROOT/'slides/baseline_algo/output/B/pose1+3'


def draw(comparison=None):
    if comparison is None:
        comparison=json.loads((HERE/'pressure_data/comparison_mg.json').read_text())
    row=next(r for r in comparison['baseline'] if r['pose']=='pose_3')
    placement=json.loads((GROUP/'step4/data/report.json').read_text())['placement']
    basis=np.array(placement['bases'][1]);offset=np.array(placement['offsets'][1])
    mesh=trimesh.load(GROUP/'step4/shape.obj',force='mesh',process=False)
    mesh.vertices=(mesh.vertices-offset)@basis.T
    domain=json.loads((GROUP/'step4/data/source_inputs/pose_3/needs.json').read_text())
    obj=trimesh.Trimesh(domain['geometry']['vertices_m'],domain['geometry']['faces'],process=False)
    z=np.load(GROUP/'step3_scheculer/co_design_reference/contacts_pose_3.npz')
    colors=['#d99b40','#578fae','#a66fa1']
    fig=plt.figure(figsize=(16,8),facecolor='white')
    ax=fig.add_axes([.015,.13,.53,.77],projection='3d',computed_zorder=False)
    ax.add_collection3d(Poly3DCollection(mesh.triangles,facecolor='#b8c0bd',edgecolor='none',alpha=1))
    ax.add_collection3d(Poly3DCollection(obj.triangles,facecolor='#acb9c1',edgecolor='none',alpha=.18))
    for k,(a,b) in enumerate(zip(z['offsets'][:-1],z['offsets'][1:])):
        ax.add_collection3d(Poly3DCollection(z['triangles_m'][a:b],facecolor=colors[k],edgecolor='none',alpha=1))
    cloud=np.vstack([mesh.vertices,obj.vertices]);lo=cloud.min(0);hi=cloud.max(0)
    pad=.02;corners=np.array([[lo[0]-pad,lo[1]-pad,0],[hi[0]+pad,lo[1]-pad,0],[hi[0]+pad,hi[1]+pad,0],[lo[0]-pad,hi[1]+pad,0]])
    ax.add_collection3d(Poly3DCollection([corners],facecolor='#e8ebea',edgecolor='none',alpha=.5,zorder=0))
    mid=(lo+hi)/2;span=max(hi-lo)*.53
    ax.set(xlim=(mid[0]-span,mid[0]+span),ylim=(mid[1]-span,mid[1]+span),zlim=(mid[2]-span,mid[2]+span))
    ax.set_box_aspect((1,1,1));ax.view_init(elev=24,azim=-48);ax.set_axis_off()
    ax.set_title('Baseline: Pose 3',fontsize=21,pad=5)
    fig.canvas.draw()
    # Each leader terminates at that original contact's saved center.
    positions=[(.12,.32),(.83,.77),(.83,.28)]
    for k,(center,pos) in enumerate(zip(z['centers_m'],positions)):
        x,y,_=proj3d.proj_transform(*center,ax.get_proj())
        value=row['head_max_resultant_force_mg_in_minimax_solutions'][k]
        label=f"{row['head_ids'][k].split('_')[-1]}\n{value:.1f} mg"
        ax.annotate(label,xy=(x,y),xycoords='data',xytext=pos,textcoords='axes fraction',
                    ha='center',va='center',fontsize=17,color=colors[k],fontweight='bold',
                    zorder=100,
                    bbox=dict(boxstyle='round,pad=.35',facecolor='white',edgecolor=colors[k],alpha=.95),
                    arrowprops=dict(arrowstyle='-',color=colors[k],lw=2))
        ax.plot([x],[y],transform=ax.transData,marker='o',markersize=6,color=colors[k],zorder=100)
    samples=json.loads((GROUP/'step4/data/source_inputs/pose_3/samples.json').read_text())
    i=row['worst']['index']%32768
    point=np.array(samples['pt_m'][i]);force=np.array(samples['force_push_mg'][i]);force/=np.linalg.norm(force)
    start=point-.045*force
    ax.quiver(*start,*(point-start),color='#bc3745',linewidth=2,arrow_length_ratio=.22)
    px,py,_=proj3d.proj_transform(*start,ax.get_proj())
    ax.annotate('Fpush = 0.5 mg',xy=(px,py),xycoords='data',xytext=(.19,.83),textcoords='axes fraction',
                fontsize=14,color='#bc3745',ha='center',zorder=100,arrowprops=dict(arrowstyle='-',color='#bc3745',lw=1.5))
    chart=fig.add_axes([.63,.22,.31,.63])
    names=['Dock: Pose 2','Dock: tilted pose','Dock: Pose 4']+[f"Pose 3: {x.split('_')[-1]}" for x in row['head_ids']]
    # Dock force is the vector resultant of gravity and tool force, never a
    # pressure normalized to an arbitrary reference area.
    from interface_pressure import loads
    belt=json.loads((HERE/'pressure_data/report.json').read_text())
    pinned=np.load(HERE/'original_geometry.npz')
    belt_object=trimesh.Trimesh(pinned['object_vertices'],pinned['object_faces'],process=False)
    records=json.loads((HERE/'geometry_report.json').read_text())['records']
    dock_values=[]
    for k,record in enumerate(records):
        transformed=belt_object.copy().apply_transform(np.array(record['transform']))
        _,forces,_=loads(transformed,record['working_area']['face_ids'],belt['weight_N'],np.random.default_rng(20261002+k))
        dock_values.append(float(np.max(np.linalg.norm(forces+[0,0,-belt['weight_N']],axis=1))/belt['weight_N']))
    values=dock_values+row['head_max_resultant_force_mg_in_minimax_solutions']
    bars=chart.barh(names,values,color=['#eb9d30']*3+colors,height=.6)
    chart.invert_yaxis();chart.set_xlim(0,max(values)*1.21)
    chart.set_xticks([]);chart.tick_params(axis='y',length=0,labelsize=12)
    chart.spines[['top','right','bottom','left']].set_visible(False)
    for bar,value in zip(bars,values):chart.text(value+.8,bar.get_y()+bar.get_height()/2,f'{value:.0f} mg' if value>10 else f'{value:.1f} mg',va='center',fontsize=14)
    chart.set_title('Resultant force\nrelative to object weight',fontsize=17,pad=20)
    fig.text(.065,.095,'mg = object weight. Labels show vector-summed contact force, not pressure.',fontsize=12,color='#4c565c')
    fig.text(.065,.052,'Each value is its own maximum in the model. Dock: no ground sharing. Baseline: ground sharing; selected equilibrium allocations.',fontsize=10,color='#6a7479')
    path=HERE.parent/'pose3_load.png';fig.savefig(path,dpi=170);plt.close(fig)
    (HERE/'pressure_data/pose3_render.json').write_text(json.dumps(dict(pose='pose_3',
        support_mesh=str((GROUP/'step4/shape.obj').relative_to(ROOT)),
        support_task_world_transform='(fixture - offset[1]) @ basis[1].T',
        head_centers_m=z['centers_m'].tolist(),head_values_mg=row['head_max_resultant_force_mg_in_minimax_solutions'],
        dock_resultant_maxima_mg=dock_values,quantity='magnitude of vector-summed resultant force; no pressure normalization',
        labels_are_individual_maxima_not_simultaneous=True,tool_arrow_sample_index=i,
        image=str(path.relative_to(ROOT))),indent=2)+'\n')
    print(path)


if __name__=='__main__':draw()
