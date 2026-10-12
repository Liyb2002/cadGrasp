"""Draw simultaneous reactions for four seeded downward original task loads."""
from pathlib import Path
import hashlib,json
import numpy as np
import trimesh
from scipy.optimize import linprog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
GROUP=ROOT/'slides/baseline_algo/output/B/pose1+3'
COLORS=['#d39737','#528eb1','#ac74a1']


def calculate():
    inputs=GROUP/'step4/data/source_inputs/pose_3'
    d=json.loads((inputs/'needs.json').read_text());s=json.loads((inputs/'samples.json').read_text())
    mesh=trimesh.Trimesh(d['geometry']['vertices_m'],d['geometry']['faces'],process=False)
    com=np.array(d['frame']['moment_origin_m'])
    contact_path=GROUP/'step3_scheculer/co_design_reference/contacts_pose_3.npz'
    z=np.load(contact_path)
    setup_path=ROOT/'objects/B/history/before_compatible_poses_860a4233e74b/tasks/pose_3/setup.npz'
    floor=np.load(setup_path)['floor_contact_m']
    groups=[]
    for a,b in zip(z['offsets'][:-1],z['offsets'][1:]):
        points=z['triangles_m'][a:b].reshape(-1,3)
        normal=np.repeat(-mesh.face_normals[z['source_faces'][a:b]],3,axis=0)
        raw=np.c_[normal,np.cross(points-com,normal)]
        keep=np.unique(np.round(raw,12),axis=0,return_index=True)[1]
        groups.append(raw[np.sort(keep)])
    head=np.vstack(groups);rays=np.array([[64,0,1],[-64,0,1],[0,64,1],[0,-64,1.]])
    cols=np.vstack([head,np.c_[rays,np.cross(floor-com,rays)]])
    n=len(cols);offsets=np.cumsum([0]+[len(g) for g in groups]);scale=np.r_[np.ones(3),np.full(3,1/mesh.extents.max())]
    eq=np.c_[cols.T*scale[:,None],np.zeros(6)]
    ub=np.zeros((4,n+1));ub[0,:len(head)]=-head[:,2]
    for k,(a,b) in enumerate(zip(offsets[:-1],offsets[1:])):ub[k+1,a:b]=1;ub[k+1,-1]=-1
    force=np.array(s['force_push_mg']);eligible=np.flatnonzero(force[:,2]<0)
    selected=np.random.default_rng(20261002).choice(eligible,4,replace=False)
    records=[]
    for i in selected:
        target=np.array(s['need_wrench'][i]);fit=linprog(np.r_[np.zeros(n),1],A_eq=eq,b_eq=target*scale,A_ub=ub,b_ub=np.zeros(4),bounds=(0,None),method='highs')
        if not fit.success:raise RuntimeError(fit.message)
        x=fit.x;wrench=np.array([g.T@x[a:b] for g,a,b in zip(groups,offsets[:-1],offsets[1:])])
        ground=rays.T@x[len(head):n]
        balance=wrench[:,:3].sum(0)+ground+force[i]+[0,0,-1]
        assert np.max(abs(balance))<1e-7
        assert np.max(abs(cols.T@x[:n]-target))<1e-7
        records.append(dict(sample_index=int(i),tool_point_m=s['pt_m'][i],tool_force_on_object_mg=force[i].tolist(),
            head_ids=z['candidate_ids'].tolist(),head_forces_on_object_mg=wrench[:,:3].tolist(),
            head_forces_on_head_mg=(-wrench[:,:3]).tolist(),head_resultant_magnitudes_mg=np.linalg.norm(wrench[:,:3],axis=1).tolist(),
            head_wrenches_about_object_com_mg_mgm=wrench.tolist(),ground_force_on_object_mg=ground.tolist(),
            gravity_force_on_object_mg=[0,0,-1],force_balance_residual_mg=balance.tolist(),
            wrench_residual=float(np.max(abs(cols.T@x[:n]-target)))))
        print('sample',i,'push',np.linalg.norm(force[i]),'heads',np.linalg.norm(wrench[:,:3],axis=1),'ground',ground,flush=True)
    paths=[inputs/'needs.json',inputs/'samples.json',contact_path,setup_path,Path(__file__)]
    report=dict(pose='pose_3',seed=20261002,selection='Four original sampled loads with negative vertical tool-force component; no maximum-magnitude rescaling.',
        allocation='Minimize largest head normal-force sum with original frictionless head normals, mu=64 floor, shared no-uplift; one feasible optimum, not a unique elastic reaction.',
        vector_convention='All plotted arrows are forces ON THE OBJECT; force received by each head is opposite, with identical magnitude.',
        head_centers_m=z['centers_m'].tolist(),floor_point_m=floor.tolist(),object_com_m=com.tolist(),records=records,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    out=HERE/'pressure_data/random_loads.json';out.write_text(json.dumps(report,indent=2)+'\n')
    return report,mesh,z


def render(report,obj,z):
    placement=json.loads((GROUP/'step4/data/report.json').read_text())['placement']
    basis=np.array(placement['bases'][1]);offset=np.array(placement['offsets'][1])
    support=trimesh.load(GROUP/'step4/shape.obj',force='mesh',process=False)
    support.vertices=(support.vertices-offset)@basis.T
    centers=np.array(report['head_centers_m']);floor=np.array(report['floor_point_m']);com=np.array(report['object_com_m'])
    fig=plt.figure(figsize=(17,13),facecolor='white')
    fig.suptitle('Pose 3: four random downward pushes',fontsize=25,y=.974)
    fig.text(.5,.937,'Each panel is ONE simultaneous load case. mg = object weight. All arrows are forces ON the object.',ha='center',fontsize=13,color='#555555')
    for k,row in enumerate(report['records']):
        ax=fig.add_subplot(2,2,k+1,projection='3d',computed_zorder=False)
        ax.add_collection3d(Poly3DCollection(support.triangles,facecolor='#b9c2be',edgecolor='none'))
        ax.add_collection3d(Poly3DCollection(obj.triangles,facecolor='#a8b7c2',edgecolor='none',alpha=.17))
        for j,(a,b) in enumerate(zip(z['offsets'][:-1],z['offsets'][1:])):
            ax.add_collection3d(Poly3DCollection(z['triangles_m'][a:b],facecolor=COLORS[j],edgecolor='none'))
        cloud=np.vstack([support.vertices,obj.vertices]);lo=cloud.min(0);hi=cloud.max(0);mid=(lo+hi)/2;span=max(hi-lo)*.7
        ax.set(xlim=(mid[0]-span,mid[0]+span),ylim=(mid[1]-span,mid[1]+span),zlim=(mid[2]-span,mid[2]+span))
        ax.set_box_aspect((1,1,1));ax.view_init(elev=24,azim=-48);ax.set_axis_off()
        ax.set_title(f"Sample {row['sample_index']}",fontsize=16,pad=-4)
        arrows=[(np.array(row['tool_point_m']),np.array(row['tool_force_on_object_mg']),'#bc3745','Push',(.14,.86)),
                (com,np.array([0,0,-1]),'#555555','Gravity',(.1,.61)),
                (floor,np.array(row['ground_force_on_object_mg']),'#378764','Ground',(.47,.08))]
        for j,(point,force) in enumerate(zip(centers,np.array(row['head_forces_on_object_mg']))):
            arrows.append((point,force,COLORS[j],row['head_ids'][j].split('_')[-1],[(.08,.30),(.88,.73),(.88,.30)][j]))
        for point,force,color,name,label in arrows:
            # Identical world-space arrow scale for every vector and every panel.
            delta=.028*force
            start=point-delta if name=='Push' else point
            ax.quiver(*start,*delta,color=color,linewidth=2,arrow_length_ratio=.18,zorder=60)
            endpoint=point if name=='Push' else point+delta
            x,y,_=proj3d.proj_transform(*endpoint,ax.get_proj())
            ax.annotate(f'{name}: {np.linalg.norm(force):.2f} mg',xy=(x,y),xycoords='data',xytext=label,textcoords='axes fraction',
                color=color,fontsize=12,fontweight='bold',ha='center',va='center',zorder=100,
                bbox=dict(boxstyle='round,pad=.2',fc='white',ec='none',alpha=.92),
                arrowprops=dict(arrowstyle='-',color=color,lw=1))
        ax.text2D(.5,-.02,'Force and moment balance checked',transform=ax.transAxes,ha='center',fontsize=10,color='#657079')
    fig.subplots_adjust(left=.015,right=.985,top=.90,bottom=.09,wspace=.01,hspace=.12)
    fig.text(.05,.035,'Head-force magnitudes equal the forces received by the heads; directions reverse. Arrows at head centers summarize distributed contacts.',fontsize=11,color='#657079')
    path=HERE.parent/'pose3_random_loads.png';fig.savefig(path,dpi=160);plt.close(fig);print(path,flush=True)


if __name__=='__main__':
    report,obj,z=calculate();render(report,obj,z)
