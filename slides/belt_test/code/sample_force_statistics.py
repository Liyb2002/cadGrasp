"""100-load resultant statistics, with shared grounded LP assumptions.

Also evaluate the hypothetical locked belt dock on exactly the same baseline
Pose 3 geometry/loads. Historical three-dock poses retain their own task patches.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import trimesh
from scipy.optimize import linprog
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
from interface_pressure import contact_model

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
GROUP=ROOT/'slides/baseline_algo/output/B/pose1+3'
COUNT=100
SEED=20261002
COLORS=['#d39737','#528eb1','#ac74a1']


def problem(mesh,com,floor,groups):
    head=np.vstack(groups)
    rays=np.array([[64,0,1],[-64,0,1],[0,64,1],[0,-64,1.]])
    cols=np.vstack([head,np.c_[rays,np.cross(floor-com,rays)]])
    n=len(cols);offsets=np.cumsum([0]+[len(g) for g in groups])
    scale=np.r_[np.ones(3),np.full(3,1/mesh.extents.max())]
    eq=np.c_[cols.T*scale[:,None],np.zeros(6)]
    ub=np.zeros((1+len(groups),n+1));ub[0,:len(head)]=-head[:,2]
    for k,(a,b) in enumerate(zip(offsets[:-1],offsets[1:])):ub[k+1,a:b]=1;ub[k+1,-1]=-1
    def solve(point,force):
        need=np.r_[-(force+[0,0,-1]),-np.cross(point-com,force)]
        fit=linprog(np.r_[np.zeros(n),1],A_eq=eq,b_eq=need*scale,A_ub=ub,b_ub=np.zeros(len(ub)),bounds=(0,None),method='highs')
        if not fit.success:
            if fit.status!=2:raise RuntimeError(fit.message)
            return dict(feasible=False,solver_message=fit.message)
        x=fit.x
        w=np.array([g.T@x[a:b] for g,a,b in zip(groups,offsets[:-1],offsets[1:])])
        ground=rays.T@x[len(head):n]
        residual=cols.T@x[:n]-need
        assert np.max(abs(residual))<1e-7
        assert w[:,2].sum()>=-1e-7
        return dict(feasible=True,group_force_on_object_mg=w[:,:3].tolist(),
                    group_force_on_support_mg=(-w[:,:3]).tolist(),
                    group_resultant_mg=np.linalg.norm(w[:,:3],axis=1).tolist(),
                    group_normal_force_sum_mg=[float(sum(x[a:b])) for a,b in zip(offsets[:-1],offsets[1:])],
                    ground_force_on_object_mg=ground.tolist(),wrench_residual=float(max(abs(residual))),
                    head_wrenches_about_com=w.tolist())
    return solve


def summarize(records,names):
    good=[r for r in records if r['solution']['feasible']]
    stats=[]
    if good:
        values=np.array([r['solution']['group_resultant_mg'] for r in good])
        for k,name in enumerate(names):
            index=int(np.argmax(values[:,k]))
            stats.append(dict(name=name,average_mg=float(np.mean(values[:,k])),worst_mg=float(values[index,k]),
                              worst_sample_id=good[index]['sample_id']))
    return dict(count=len(records),feasible_count=len(good),infeasible_count=len(records)-len(good),
                statistics_over='feasible samples only; infeasible count shown explicitly',groups=stats)


def run_loads(solve,points,forces,ids,names):
    records=[dict(sample_id=int(i),tool_point_world_m=p.tolist(),tool_force_mg=f.tolist(),solution=solve(p,f))
             for p,f,i in zip(points,forces,ids)]
    return dict(**summarize(records,names),records=records)


def sampled_downward(mesh,work_ids,seed):
    """Uniform area/cone/magnitude proposals, conditioned on Fz<0.

    No cone-boundary overweighting, extrema rescaling or selection by reaction.
    Tool self-occlusion is not tested for these historical illustrative tasks.
    """
    rng=np.random.default_rng(seed);points=[];forces=[];ids=[];proposal=0
    faces=np.asarray(work_ids);prob=mesh.area_faces[faces]/mesh.area_faces[faces].sum()
    while len(points)<COUNT:
        chosen=rng.choice(faces,256,p=prob);bary=rng.dirichlet([1,1,1],256)
        q=np.einsum('ni,nij->nj',bary,mesh.triangles[chosen])
        normal=-mesh.face_normals[chosen]
        tangent=np.cross(normal,np.eye(3)[np.argmin(abs(normal),axis=1)]);tangent/=np.linalg.norm(tangent,axis=1)[:,None]
        other=np.cross(normal,tangent)
        theta=np.arccos(rng.uniform(np.cos(np.pi/6),1,256));phi=rng.uniform(0,2*np.pi,256)
        direction=np.cos(theta)[:,None]*normal+np.sin(theta)[:,None]*(np.cos(phi)[:,None]*tangent+np.sin(phi)[:,None]*other)
        force=direction*rng.uniform(0,.5,256)[:,None]
        for j in np.flatnonzero(force[:,2]<0):
            if len(points)==COUNT:break
            points.append(q[j]);forces.append(force[j]);ids.append(proposal+int(j))
        proposal+=256
        if proposal>100000:raise RuntimeError('Insufficient downward task directions')
    return np.array(points),np.array(forces),np.array(ids)


def dock_problem(mesh,port,basis,floor,locked=True):
    _,tiles=contact_model(locked=locked)
    points=np.array(tiles['points_m'])@basis.T+port
    normals=np.array(tiles['normals'])@basis.T
    com=mesh.center_mass
    w=np.c_[normals,np.cross(points-com,normals)]
    return problem(mesh,com,floor,[w])


def floor_point(mesh):
    return np.mean(mesh.vertices[mesh.vertices[:,2]<=mesh.vertices[:,2].min()+1e-8],axis=0)


def calculate():
    inputs=GROUP/'step4/data/source_inputs/pose_3'
    d=json.loads((inputs/'needs.json').read_text());s=json.loads((inputs/'samples.json').read_text())
    mesh=trimesh.Trimesh(d['geometry']['vertices_m'],d['geometry']['faces'],process=False)
    com=np.array(d['frame']['moment_origin_m'])
    cp=GROUP/'step3_scheculer/co_design_reference/contacts_pose_3.npz';z=np.load(cp)
    sp=ROOT/'objects/B/history/before_compatible_poses_860a4233e74b/tasks/pose_3/setup.npz'
    floor=np.load(sp)['floor_contact_m'];groups=[]
    for a,b in zip(z['offsets'][:-1],z['offsets'][1:]):
        points=z['triangles_m'][a:b].reshape(-1,3)
        normal=np.repeat(-mesh.face_normals[z['source_faces'][a:b]],3,axis=0)
        raw=np.c_[normal,np.cross(points-com,normal)]
        keep=np.unique(np.round(raw,12),axis=0,return_index=True)[1];groups.append(raw[np.sort(keep)])
    forces=np.array(s['force_push_mg']);eligible=np.flatnonzero(forces[:,2]<0)
    selected=np.random.default_rng(SEED).choice(eligible,COUNT,replace=False)
    q=np.array(s['pt_m'])[selected];f=forces[selected]
    baseline=run_loads(problem(mesh,com,floor,groups),q,f,selected,z['candidate_ids'].tolist())
    pinned=np.load(HERE/'original_geometry.npz')
    obj=trimesh.Trimesh(pinned['object_vertices'],pinned['object_faces'],process=False)
    # Rigidly register the historical belt on exactly the baseline object pose.
    frame_path=ROOT/'codes/simulation/shape/B/pose_2/load_domain.json'
    frame=np.array(json.loads(frame_path.read_text())['frame']['T_world_mesh'])
    align=np.array(d['frame']['T_world_mesh'])@np.linalg.inv(frame)
    transformed=obj.vertices@align[:3,:3].T+align[:3,3]
    # The mesh may have additional subdivisions. Check that all pinned original
    # vertices occur in the task mesh, as well as volume, area and COM agreement.
    old=trimesh.Trimesh(transformed,obj.faces,process=False)
    vertex_error=float(cKDTree(mesh.vertices).query(transformed)[0].max())
    assert vertex_error<1e-8
    assert abs(old.volume-mesh.volume)<1e-12
    assert abs(old.area-mesh.area)<1e-12
    assert np.linalg.norm(old.center_mass-com)<1e-9
    port=align[:3,:3]@pinned['port']+align[:3,3];basis=align[:3,:3]@pinned['basis']
    matched=run_loads(dock_problem(mesh,port,basis,floor),q,f,selected,['Dock on same Pose 3'])
    unlocked=run_loads(dock_problem(mesh,port,basis,floor,locked=False),q,f,selected,['Original unlocked dock'])
    records=json.loads((HERE/'geometry_report.json').read_text())['records'];historical=[]
    for k,record in enumerate(records):
        T=np.array(record['transform']);task=obj.copy().apply_transform(T)
        port=T[:3,:3]@pinned['port']+T[:3,3];basis=T[:3,:3]@pinned['basis']
        tq,tf,ids=sampled_downward(task,record['working_area']['face_ids'],SEED+k)
        result=run_loads(dock_problem(task,port,basis,floor_point(task)),tq,tf,ids,[record['pose']])
        result.update(pose=record['pose'],pose_kind=record['pose_kind'],floor_point_m=floor_point(task).tolist(),seed=SEED+k)
        historical.append(result)
    sources=[inputs/'needs.json',inputs/'samples.json',cp,sp,HERE/'original_geometry.npz',HERE/'geometry_report.json',frame_path,Path(__file__),HERE/'interface_pressure.py']
    report=dict(count=COUNT,seed=SEED,quantity='Norm of vector-summed force received by each whole contact group, in object weights mg. No pressure conversion.',
        selection='Baseline: 100 original task samples without replacement, conditioned on Fz<0. Matched dock: identical points and force vectors. Historical docks: 100 independent area/cone/magnitude draws conditioned on Fz<0, not tool-ray screened.',
        magnitude='Original magnitudes / uniform [0,0.5]mg for newly drawn tasks; not all at 0.5mg.',
        ground='Object ground contact included in EVERY solve, same four-ray mu=64 model and shared no-uplift. Baseline/matched dock share exact archived floor point. Historical docks use mean of lowest mesh vertices as one contact point.',
        allocation='Same LP: minimize largest group normal-force sum; then measure vector resultant. Baseline has three groups, dock one group. Reaction allocation is not unique and does not predict elastic sharing.',
        dock='Rigidly attached massless belt, hypothetical ideal zero-clearance locked interface (extra retainer), infinite contact capacity; base force/torque balance and real locking mechanics are not certified.',
        worst_definition='Largest of the 100 draws, not a continuous-domain maximum.',
        baseline=baseline,matched_dock=matched,matched_unlocked_diagnostic=unlocked,historical_docks=historical,
        matched_object_registration=dict(transform=align.tolist(),center_mass_error_m=float(np.linalg.norm(old.center_mass-com)),object_volume_error_m3=float(abs(old.volume-mesh.volume)),pinned_vertex_match_error_m=vertex_error),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    out=HERE/'pressure_data/statistics_100.json';out.write_text(json.dumps(report,indent=2)+'\n')
    for label,result in [('baseline',baseline),('matched dock',matched)]+[(r['pose'],r) for r in historical]:
        print(label,result['feasible_count'],'/',COUNT,result['groups'],flush=True)
    return report,mesh,z


def render(report,obj,z):
    placement=json.loads((GROUP/'step4/data/report.json').read_text())['placement'];basis=np.array(placement['bases'][1]);offset=np.array(placement['offsets'][1])
    support=trimesh.load(GROUP/'step4/shape.obj',force='mesh',process=False);support.vertices=(support.vertices-offset)@basis.T
    fig=plt.figure(figsize=(17,8),facecolor='white');ax=fig.add_axes([.01,.17,.49,.73],projection='3d',computed_zorder=False)
    ax.add_collection3d(Poly3DCollection(support.triangles,facecolor='#b9c2be',edgecolor='none'))
    ax.add_collection3d(Poly3DCollection(obj.triangles,facecolor='#a8b7c2',edgecolor='none',alpha=.17))
    for k,(a,b) in enumerate(zip(z['offsets'][:-1],z['offsets'][1:])):ax.add_collection3d(Poly3DCollection(z['triangles_m'][a:b],facecolor=COLORS[k],edgecolor='none'))
    cloud=np.vstack([support.vertices,obj.vertices]);lo=cloud.min(0);hi=cloud.max(0);mid=(lo+hi)/2;span=max(hi-lo)*.56
    ax.set(xlim=(mid[0]-span,mid[0]+span),ylim=(mid[1]-span,mid[1]+span),zlim=(mid[2]-span,mid[2]+span));ax.set_box_aspect((1,1,1));ax.view_init(elev=24,azim=-48);ax.set_axis_off()
    ax.set_title('Pose 3: 100 sampled pushes',fontsize=21,pad=0)
    for k,(center,stats,pos) in enumerate(zip(z['centers_m'],report['baseline']['groups'],[(.08,.30),(.85,.77),(.85,.28)])):
        x,y,_=proj3d.proj_transform(*center,ax.get_proj())
        ax.annotate(f"{stats['name'].split('_')[-1]}\nAvg {stats['average_mg']:.2f} mg\nWorst {stats['worst_mg']:.2f} mg",xy=(x,y),xycoords='data',xytext=pos,textcoords='axes fraction',ha='center',va='center',fontsize=15,color=COLORS[k],fontweight='bold',zorder=100,bbox=dict(boxstyle='round,pad=.3',fc='white',ec=COLORS[k],alpha=.96),arrowprops=dict(arrowstyle='-',color=COLORS[k],lw=2))
    rows=report['baseline']['groups']+report['matched_dock']['groups']+[r['groups'][0] for r in report['historical_docks']]
    names=['Head '+r['name'].split('_')[-1] for r in report['baseline']['groups']]+['Dock: SAME 100 Pose 3 loads','Dock: historical Pose 2','Dock: historical tilted pose','Dock: historical Pose 4']
    chart=fig.add_axes([.67,.21,.26,.65]);y=np.arange(len(rows));avg=[r['average_mg'] for r in rows];worst=[r['worst_mg'] for r in rows]
    chart.barh(y-.17,avg,height=.3,color='#8bb7cc',label='Average');chart.barh(y+.17,worst,height=.3,color='#d99543',label='Worst of 100')
    chart.set_yticks(y,names);chart.tick_params(axis='y',labelsize=11,length=0);chart.invert_yaxis();chart.spines[['top','right','left','bottom']].set_visible(False);chart.set_xticks([])
    for k,(a,w) in enumerate(zip(avg,worst)):
        chart.text(a+.025,k-.17,f'{a:.2f} mg',va='center',fontsize=10);chart.text(w+.025,k+.17,f'{w:.2f} mg',va='center',fontsize=10)
    chart.set_xlim(0,max(worst)*1.25);chart.set_title('Resultant force / object weight',fontsize=17,pad=17);chart.legend(loc='upper center',bbox_to_anchor=(.5,-.025),frameon=False,ncol=2,fontsize=10)
    fig.text(.05,.11,'100 random loads with a downward component; magnitudes retained (up to 0.5 mg). Average = mean magnitude; worst = largest sample.',fontsize=11,color='#555555')
    fig.text(.05,.065,'Object-ground support, friction and force-allocation algorithm included in both designs. Dock assumes an ideal lock and attached belt.',fontsize=11,color='#555555')
    fig.text(.05,.028,'The SAME-100 row shares exact baseline loads and floor point. Historical dock rows have different poses/work areas. Forces include moments in the solve.',fontsize=10,color='#6a7479')
    path=HERE.parent/'force_statistics_100.png';fig.savefig(path,dpi=170);plt.close(fig);print(path,flush=True)


if __name__=='__main__':
    report,mesh,z=calculate();render(report,mesh,z)
