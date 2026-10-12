"""Geometric witnesses for the saved, identical video-01 fixture; no force solve."""
from pathlib import Path
import json,shutil
import scene as S
from scene import np,C,G,TEMP
from scipy.spatial.transform import Rotation,Slerp
from work_access import tangent_frame

def prepare_analysis(scene):
    rows=scene.rows;Ts=[row['fixture_transform'] for row in rows]
    vertices=np.asarray(scene.final.vertices)
    minima=[float(C.F.transform_points(vertices,T)[:,2].min()) for T in Ts]
    transitions=[];minimum=min(minima)
    for start,end in zip(Ts,Ts[1:]+Ts[:1]):
        relative=Rotation.from_matrix(start[:3,:3].T@end[:3,:3]).as_rotvec()
        theta=float(np.linalg.norm(relative));k=relative/theta
        n=start[2,:3];parallel=np.outer(vertices@k,k)
        a=(vertices-parallel)@n;b=np.cross(k,vertices)@n
        c=parallel@n+start[2,3];delta=end[2,3]-start[2,3]
        low=np.minimum(a+c,a*np.cos(theta)+b*np.sin(theta)+c+delta)
        radius=np.sqrt(a*a+b*b);q=np.full(len(radius),2.)
        usable=radius>1e-14;q[usable]=-delta/(theta*radius[usable])
        valid=np.abs(q)<=1.;angle=np.arccos(np.clip(q,-1,1));phase=np.arctan2(a,b)
        for sign in (-1,1):
            for turn in range(-2,3):
                u=(sign*angle+2*np.pi*turn-phase)/theta
                mask=valid&(u>=0)&(u<=1)
                z=a*np.cos(theta*u)+b*np.sin(theta*u)+c+delta*u
                low[mask]=np.minimum(low[mask],z[mask])
        edge_min=float(low.min());minimum=min(minimum,edge_min)
        transitions.append(dict(start=start.tolist(),end=end.tolist(),minimum_z_m=edge_min))
    tasks,body=S.D.load_tasks(scene.data,scene.inp)
    # Interior points of the actual outer-rim cells, not fitted demand points.
    witnesses=[]
    for ring in scene.rings:
        for i in range(len(ring.inner)):
            j=(i+1)%len(ring.inner)
            corners=np.array([ring.inner[i],ring.inner[j],ring.outer[j],ring.outer[i]])
            center=corners.mean(0)
            for xy in np.vstack([center,.99*corners+.01*center]):
                for height in [.0005,S.HEIGHT/2,S.HEIGHT-.0005]:
                    witnesses.append(C.F.transform_points(np.array([[*xy,height]]),ring.inverse)[0])
        for f in [.1,.3,.5,.7,.9]:witnesses.append((1-f)*ring.start+f*ring.end)
    witnesses=np.asarray(witnesses)
    works=[]
    for index,(task,row) in enumerate(zip(tasks,rows)):
        placement=np.linalg.inv(row['fixture_transform'])@row['object_transform']
        local=C.F.transform_points(witnesses,np.linalg.inv(placement))
        ids=np.asarray(task.domain.work_ids,int);alpha=float(task.domain.data['load']['cone_half_deg'])
        choices=[]
        for face in ids:
            tri=body.triangles[face];normal=body.face_normals[face]
            nearest=S.trimesh.triangles.closest_point(np.broadcast_to(tri,(len(local),3,3)).copy(),local)
            vector=local-nearest;distance=np.linalg.norm(vector,axis=1)
            axial=vector@normal
            angles=np.degrees(np.arccos(np.clip(axial/np.maximum(distance,1e-20),-1,1)))
            valid=(axial>.003)&(angles<alpha-.25)
            for j in np.flatnonzero(valid):
                choices.append((float(distance[j]),float(angles[j]),int(face),int(j),nearest[j],vector[j]/distance[j]))
        if not choices:
            work=dict(pose=row['pose'],work_face_count=len(ids),half_angle_deg=alpha,
                      selected_face=None,true_rim_ray_found=False,placement=placement.tolist())
            np.savez(TEMP/f'work_{index}.npz',triangles=body.triangles[ids],normals=body.face_normals[ids],
                     face_ids=ids,placement=placement)
            works.append(work)
            continue
        # A clearly visible ray with a strict angular margin, ending inside material.
        best=max(choices,key=lambda x:min(x[0],.055)-.0004*x[1])
        distance,angle,face,j,root,direction=best
        ray_root=C.F.transform_points(root[None],placement)[0]
        ray_direction=placement[:3,:3]@direction
        assert np.allclose(ray_root+distance*ray_direction,witnesses[j],atol=1e-12)
        tri=body.triangles[face];normal=body.face_normals[face];frame=tangent_frame(normal)
        phases=np.arange(32)*2*np.pi/32
        ring=np.tan(np.radians(alpha))*(np.cos(phases)[:,None]*frame[:,0]+np.sin(phases)[:,None]*frame[:,1])
        cap=max(.02,float(((C.F.transform_points(vertices,np.linalg.inv(placement))-tri[0])@normal).max())+.005)
        cloud=np.vstack([tri,(tri[:,None]+cap*(normal+ring)[None]).reshape(-1,3)])
        cone=S.trimesh.convex.convex_hull(cloud)
        common_cone=G.transform(G.solid(cone),placement)
        collision=scene.solid^common_cone
        assert G.volume(collision)>1e-14
        G.unpack(collision).export(TEMP/f'work_collision_{index}.obj')
        work=dict(pose=row['pose'],work_face_count=len(ids),half_angle_deg=alpha,selected_face=face,true_rim_ray_found=True,
            selected_ray_root_m=ray_root.tolist(),selected_ray_direction=ray_direction.tolist(),
            selected_ray_material_distance_m=distance,selected_ray_angle_deg=angle,
            strict_selected_cone_overlap_cm3=G.volume(collision)*1e6,placement=placement.tolist())
        np.savez(TEMP/f'work_{index}.npz',triangles=body.triangles[ids],normals=body.face_normals[ids],face_ids=ids,
                 placement=placement,ray_root=ray_root,ray_direction=ray_direction,witness=witnesses[j])
        works.append(work)
    answer=dict(support_geometry_sha256=scene.support_hash,poses=[row['pose'] for row in rows],
        installed_min_z_mm=[m*1000 for m in minima],installed_lift_to_floor_m=[max(0.,-m) for m in minima],
        continuous_geodesic_cycle_min_z_m=minimum,safe_rotation_lift_m=max(0.,-minimum)+.002,
        transitions=transitions,work=works,scope='actual fixed video fixture; no arbitrary-base impossibility claim')
    C.save(TEMP/'geometry.json',answer)
    print(json.dumps({k:answer[k] for k in ['installed_min_z_mm','safe_rotation_lift_m']}),flush=True)
    for work in works:print(json.dumps({k:work.get(k) for k in ['pose','selected_face','selected_ray_angle_deg','strict_selected_cone_overlap_cm3']}),flush=True)
    return answer

if __name__=='__main__':
    scene=S.Scene();directory=TEMP/'execution_analysis';sources=C.freeze(directory)
    for path in [Path(__file__),Path(S.__file__)]:shutil.copy2(path,directory/path.name)
    prepare_analysis(scene);scene.verify()
    for path,h in sources.items():assert C.digest(S.ROOT/path)==C.digest(directory/'executed_sources'/path)==h
    print('Exact shared fixture and original inputs preserved.',flush=True)
