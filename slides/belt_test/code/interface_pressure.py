"""Reproducible nominal dock-bearing screen of the pinned three-pose device.

This is a rigid, frictionless, ideal-snug distributed-contact calculation,
not FEA or a material failure test. The real 0.5 mm clearance is not closed
by this model. A hypothetical retaining face is explicitly evaluated too.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import trimesh
from scipy.optimize import linprog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / 'pressure_data'
MASS_FILE = ROOT / 'codes/simulation/shape/B/pose_2/object_meta.json'
G = 9.81


def contact_model(nlong=8, nwide=4, locked=False):
    """Uniform pressure on rectangular tiles, point resultant at tile centroid.

    Unknowns are tile-average pressures (kPa), bounded by a common t.
    A rigid-body LP minimizes t. Wall footprints are the peg bearing widths,
    with ideal zero clearance; actual sleeve overlap is 21 mm (z=-26..-5).
    """
    points, normals, areas, labels = [], [], [], []
    def face(axis, position, spans, normal, name):
        axes = [k for k in range(3) if k != axis]
        counts = [nlong if k == 2 else nwide for k in axes]
        edges = [np.linspace(a,b,n+1) for (a,b),n in zip(spans,counts)]
        for i in range(counts[0]):
            for j in range(counts[1]):
                p = np.zeros(3); p[axis] = position
                p[axes[0]] = (edges[0][i]+edges[0][i+1])/2
                p[axes[1]] = (edges[1][j]+edges[1][j+1])/2
                points.append(p); normals.append(normal)
                areas.append((edges[0][i+1]-edges[0][i])*(edges[1][j+1]-edges[1][j]))
                labels.append(name)
    for sign in (-1,1):
        face(0,sign*.009,[(-.004,.004),(-.026,-.005)],[-sign,0,0],f'x{sign:+d}')
        face(1,sign*.004,[(-.009,.009),(-.026,-.005)],[0,-sign,0],f'y{sign:+d}')
    face(2,-.026,[(-.009,.009),(-.004,.004)],[0,0,1],'end_stop')
    if locked:
        # Hypothetical full-area retainer: absent from the current apparatus.
        face(2,0,[(-.009,.009),(-.004,.004)],[0,0,-1],'hypothetical_retainer')
    points, normals, areas = map(np.asarray,(points,normals,areas))
    forces = normals*areas[:,None]*1000  # one kPa per variable -> N
    columns = np.c_[forces,np.cross(points,forces)].T
    scale = np.array([1,1,1,100,100,100.])
    eq = np.c_[columns*scale[:,None],np.zeros(6)]
    ub = np.c_[np.eye(len(points)),-np.ones(len(points))]
    objective = np.r_[np.zeros(len(points)),1.]
    def solve(need, economical=False):
        fit = linprog(objective,A_eq=eq,b_eq=need*scale,A_ub=ub,
                      b_ub=np.zeros(len(points)),bounds=(0,None),method='highs')
        if not fit.success:
            if fit.status != 2: raise RuntimeError(fit.message)
            return None
        optimum=float(fit.x[-1])
        if economical:
            # Among peak-optimal solutions reduce unnecessary self-stress.
            fit=linprog(np.r_[areas*1000,0.],A_eq=eq,b_eq=need*scale,
                        bounds=[(0,optimum+1e-7)]*len(points)+[(optimum,optimum)],method='highs')
            if not fit.success:raise RuntimeError(fit.message)
        assert np.max(abs(columns@fit.x[:-1]-need)) < 1e-6
        return optimum,fit.x[:-1]
    return solve,dict(points_m=points.tolist(),normals=normals.tolist(),
                     tile_area_m2=areas.tolist(),labels=labels,
                     grid=[nlong,nwide],locked=locked)


def loads(mesh, ids, weight, rng):
    """Full 0.5mg inward pushes: every work-face centroid + 1024 area samples.

    Deterministic cone interior and boundary samples; no global-max claim.
    Tool rays/finite tool and robot reachability are not screened here.
    """
    ids = np.asarray(ids)
    sampled = rng.choice(ids,1024,p=mesh.area_faces[ids]/mesh.area_faces[ids].sum())
    bary = rng.dirichlet([1,1,1],1024)
    points = np.r_[mesh.triangles_center[ids],np.einsum('ni,nij->nj',bary,mesh.triangles[sampled])]
    faces = np.r_[ids,sampled]
    inward = -mesh.face_normals[faces]
    tangent = np.cross(inward,np.eye(3)[np.argmin(abs(inward),axis=1)])
    tangent /= np.linalg.norm(tangent,axis=1)[:,None]
    second = np.cross(inward,tangent)
    theta = np.r_[np.zeros(len(ids)),np.arccos(rng.uniform(np.cos(np.pi/6),1,1024))]
    theta[len(ids)::2] = np.pi/6
    phi = rng.uniform(0,2*np.pi,len(faces))
    direction = np.cos(theta)[:,None]*inward+np.sin(theta)[:,None]*(np.cos(phi)[:,None]*tangent+np.sin(phi)[:,None]*second)
    return points,.5*weight*direction,faces


def calculate():
    OUT.mkdir(exist_ok=True)
    z = np.load(HERE/'original_geometry.npz')
    meshes = {k:trimesh.Trimesh(z[k+'_vertices'],z[k+'_faces'],process=False)
              for k in ('object','blue','socket','peg')}
    records = json.loads((HERE/'geometry_report.json').read_text())['records']
    mass = json.loads(MASS_FILE.read_text())['mass_kg']; weight = mass*G
    assert np.isclose(meshes['object'].volume*1000,mass)
    free, _ = contact_model(); locked,tiles = contact_model(locked=True)
    rows=[]
    for k,record in enumerate(records):
        T=np.asarray(record['transform']); R=T[:3,:3]; offset=T[:3,3]
        obj=meshes['object'].copy().apply_transform(T)
        basis=R@z['basis']; port=R@z['port']+offset
        points,forces,faces=loads(obj,record['working_area']['face_ids'],weight,np.random.default_rng(20261002+k))
        gravity=np.array([0,0,-weight]); com=obj.center_mass
        external_force=forces+gravity
        external_moment=np.cross(points-port,forces)+np.cross(com-port,gravity)
        needs=-np.c_[external_force@basis,external_moment@basis]
        worst=None; failures=0; feasible_max=0.
        for i,need in enumerate(needs):
            a=free(need)
            if a is None:failures+=1
            else:feasible_max=max(feasible_max,a[0])
            result=locked(need)
            if result is None:raise RuntimeError('Ideal locked interface unexpectedly infeasible')
            if worst is None or result[0]>worst[0]:worst=(result[0],i,result[1])
        p,i,_=worst
        _,pressures=locked(needs[i],economical=True)
        fine,fine_tiles=contact_model(16,8,locked=True)
        refined=fine(needs[i]); assert refined is not None
        row=dict(pose=record['pose'],pose_kind=record['pose_kind'],samples=len(needs),
                 unlocked_infeasible_samples=failures,
                 unlocked_max_feasible_nominal_pressure_kpa=feasible_max,
                 locked_sampled_max_minimum_tile_pressure_kpa=p,
                 selected_case_refined_pressure_kpa=refined[0],
                 max_outward_axial_load_N=max(0.,float((external_force@basis)[:,2].max())),
                 worst=dict(index=i,work_face=int(faces[i]),tool_point_world_m=points[i].tolist(),
                            tool_force_world_N=forces[i].tolist(),
                            interface_reaction_local_N_Nm=needs[i].tolist(),
                            moment_magnitude_Nm=float(np.linalg.norm(needs[i,3:])),
                            tile_pressure_kpa=pressures.tolist()),
                 transform=T.tolist(),basis=basis.tolist(),port_m=port.tolist())
        rows.append(row);print(record['pose'],p,'kPa; unlocked failures',failures,'/',len(needs),flush=True)
    sources=[HERE/'original_geometry.npz',HERE/'geometry_report.json',MASS_FILE,Path(__file__)]
    report=dict(mass_kg=mass,gravity_m_s2=G,weight_N=weight,max_tool_force_N=.5*weight,
                density_assumed_kg_m3=1000,interface_dimensions_mm=dict(peg=[18,8,26],engagement=21,clearance_per_side=.5),
                scope='Object rigidly fixed to massless belt; all object gravity and tool load routed through dock; no object-ground reactions. Fixed dock/base. Nominal frictionless zero-clearance wall contact; optional hypothetical full-area retainer.',
                pressure_definition='Minimum achievable maximum tile-average pressure for each sampled wrench; sampled maximum of those minima. Not actual elastic peak pressure, global maximum, or strength certificate.',
                load_sampling='Every work-face centroid with inward normal plus 1024 seeded area-weighted surface/cone samples per pose, all at 0.5mg, cone half-angle 30 degrees. No tool-ray screening.',
                conclusion='Pressure alone does not establish failure. Current unlocked, clearance-bearing joint is not a certified locked precision interface. Material limits, preload and compliance are unspecified.',
                contact_model=tiles,records=rows,
                source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report,meshes,z


def render(report,meshes,z):
    # Reproduce the existing common-ring/post construction from shared_base.py,
    # using pinned pose transforms; no live baseline search output is imported.
    from shapely.geometry import MultiPoint,Point
    from geometry_utils import tube,resample
    scene=[]
    for row in report['records']:
        T=np.array(row['transform']);R=T[:3,:3];t=T[:3,3]
        scene.append(dict(object=meshes['object'].vertices@R.T+t,
                          socket=meshes['socket'].copy().apply_transform(T),
                          port=np.array(row['port_m']),basis=np.array(row['basis'])))
    cloud=np.vstack([c['object'] for c in scene]+[c['port'][None,:] for c in scene])
    footprint=MultiPoint(cloud[:,:2]).convex_hull.buffer(.035)
    xy=resample(footprint,96)
    base=tube(np.c_[xy,np.full(len(xy),.005)],.005,True)
    for c in scene:
        x,y,d=c['basis'].T;mount=c['port']-d*.0315-y*.004
        back=mount-d*.020+x*.025
        local=MultiPoint(np.vstack([c['object'][:,:2],c['port'][None,:2]])).convex_hull.buffer(.023)
        edge=local.exterior.interpolate(local.exterior.project(Point(back[:2])))
        foot=np.array([edge.x,edge.y,.005]);elbow=np.array([edge.x,edge.y,max(.018,back[2]-.010)])
        outer=footprint.exterior.interpolate(footprint.exterior.project(edge))
        anchor=np.array([outer.x,outer.y,.005])
        base += [c['socket']]+tube(np.array([anchor,foot,elbow,back,mount]),.0045)
    base_mesh=trimesh.util.concatenate(base)
    fig=plt.figure(figsize=(18,11),facecolor='white')
    fig.suptitle('Fixed belt + dock: interface load at |Ftool| = 0.5 mg',fontsize=23,y=.975)
    fig.text(.5,.929,f"Object mass {report['mass_kg']:.3f} kg | mg = {report['weight_N']:.2f} N | tool force = {report['max_tool_force_N']:.2f} N | peg 18 x 8 mm, engagement 21 mm",ha='center',fontsize=12,color='#555555')
    for k,row in enumerate(report['records']):
        ax=fig.add_subplot(2,3,k+1,projection='3d')
        T=np.array(row['transform']); R=T[:3,:3]; t=T[:3,3]
        transformed={key:mesh.vertices@R.T+t for key,mesh in meshes.items()}
        for key,color,alpha in [('object','#b7bdc0',.5),('blue','#2570bc',1),('socket','#eb9d30',1)]:
            mesh=meshes[key];verts=transformed[key]
            ax.add_collection3d(Poly3DCollection(verts[mesh.faces],facecolor=color,edgecolor='none',alpha=alpha))
        ax.add_collection3d(Poly3DCollection(base_mesh.triangles,facecolor='#eb9d30',edgecolor='none'))
        ids=json.loads((HERE/'geometry_report.json').read_text())['records'][k]['working_area']['face_ids']
        ax.add_collection3d(Poly3DCollection(transformed['object'][meshes['object'].faces[ids]],facecolor='#4eb86f',edgecolor='none',alpha=.65))
        q=np.array(row['worst']['tool_point_world_m']);f=np.array(row['worst']['tool_force_world_N'])
        start=q-.045*f/np.linalg.norm(f)
        ax.quiver(*start,*(q-start),color='#ba303d',arrow_length_ratio=.25,linewidth=2)
        port=np.array(row['port_m']);ax.scatter(*port,color='#ba303d',s=35)
        cloud=np.vstack(list(transformed.values())+[base_mesh.vertices]);mid=(cloud.min(0)+cloud.max(0))/2;span=np.ptp(cloud,axis=0).max()*.54
        ax.set(xlim=(mid[0]-span,mid[0]+span),ylim=(mid[1]-span,mid[1]+span),zlim=(mid[2]-span,mid[2]+span))
        ax.set_box_aspect((1,1,1));ax.view_init(elev=22,azim=-50);ax.set_axis_off()
        title=['Pose 2','Pose 2 + 25 deg tilt (illustrative)','Pose 4'][k]
        ax.set_title(title,fontsize=15,pad=0)
        ax.text2D(.5,-.025,f"With ideal retainer: {row['locked_sampled_max_minimum_tile_pressure_kpa']:.1f} kPa\nSelected-case refined grid: {row['selected_case_refined_pressure_kpa']:.1f} kPa",transform=ax.transAxes,ha='center',fontsize=13,color='#174973')
        ax2=fig.add_subplot(2,3,k+4)
        labels=report['contact_model']['labels'];press=np.array(row['worst']['tile_pressure_kpa'])
        names=['x-1','x+1','y-1','y+1','end_stop','hypothetical_retainer']
        vals=[np.mean(press[np.array(labels)==name]) for name in names]
        ax2.barh(range(6),vals,color=['#eb9d30']*5+['#a8b4c4'])
        ax2.set_yticks(range(6),['Wall X-','Wall X+','Wall Y-','Wall Y+','End stop','Added retainer'])
        ax2.invert_yaxis();ax2.set_xlabel('Mean pressure over nominal face (kPa)')
        ax2.spines[['top','right']].set_visible(False)
        ax2.set_title(f"Interface moment: {row['worst']['moment_magnitude_Nm']:.3f} N m",fontsize=12)
        ax2.text(0,-.25,f"Original unlocked joint: {row['unlocked_infeasible_samples']}/{row['samples']} sampled loads infeasible\nMax outward axial demand: {row['max_outward_axial_load_N']:.2f} N",transform=ax2.transAxes,fontsize=11,color='#ba303d')
    fig.subplots_adjust(left=.065,right=.975,top=.88,bottom=.22,hspace=.48,wspace=.4)
    fig.text(.055,.095,'Model: fixed object + massless belt; dock carries all gravity and tool load; ground load-sharing omitted; base assumed fixed.',fontsize=11,color='#555555')
    fig.text(.055,.067,'Pressure screen assumes zero clearance and an added full-area retainer. Existing joint has 0.5 mm side clearance and no lock.',fontsize=11,color='#555555')
    fig.text(.055,.039,'These are nominal sampled demands, not actual peak contact pressures. No material limit is specified: pressure-induced failure is not established.',fontsize=11,color='#ba303d')
    path=HERE.parent/'interface_pressure_three_poses.png';fig.savefig(path,dpi=170);plt.close(fig);print(path,flush=True)


if __name__=='__main__':
    report,meshes,z=calculate();render(report,meshes,z)
