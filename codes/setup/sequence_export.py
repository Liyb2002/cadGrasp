"""Export verified sequence inputs, including seeded connected work patches."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zlib

import numpy as np
import trimesh

ROOT=Path(__file__).resolve().parents[2]
WORK_AREA_FRACTION=(.06,.10)
sys.path.insert(0,str(ROOT/'slides/setup/poses'))
import big_tip as B


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


class WorkRegions:
    def __init__(self,raw):
        self.mesh,self.rounds=B.refine(raw)
        # Large source triangles may individually exceed the new work band,
        # even when the historical refinement's face-count budget is reached.
        while self.mesh.area_faces.max()/self.mesh.area>WORK_AREA_FRACTION[0]:
            self.mesh=self.mesh.subdivide()
            self.rounds+=1
        m=self.mesh
        self.visible=~m.ray.intersects_any(m.triangles_center+1e-5*m.face_normals,m.face_normals)
        self.adj=[[] for _ in m.faces];self.weights=[[] for _ in m.faces]
        pairs=m.face_adjacency
        distance=np.linalg.norm(m.triangles_center[pairs[:,0]]-m.triangles_center[pairs[:,1]],axis=1)
        for (i,j),d in zip(pairs,distance):
            self.adj[i].append(j);self.adj[j].append(i)
            self.weights[i].append(d);self.weights[j].append(d)

    def feasible(self,T):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        m=self.mesh
        world_z=m.vertices@T[2,:3]+T[2,3]
        eligible=self.visible & ((m.face_normals@T[2,:3])>.35)
        eligible &= world_z[m.faces].min(axis=1)>.0015
        pairs=m.face_adjacency
        pairs=pairs[eligible[pairs].all(axis=1)]
        graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(m.faces),len(m.faces)))
        _,labels=connected_components(graph,directed=False)
        sums=np.bincount(labels,weights=m.area_faces*eligible)
        return float(sums.max())>=WORK_AREA_FRACTION[0]*m.area

    def choose(self,T,key):
        m=self.mesh
        world=trimesh.transform_points(m.vertices,T)
        eligible=self.visible & ((m.face_normals@T[:3,:3].T)[:,2]>.35)
        eligible &= world[m.faces,2].min(axis=1)>.0015
        ids=np.flatnonzero(eligible)
        if not len(ids):raise ValueError(f'{key}: no work area candidates')
        rng=np.random.default_rng(20260918^zlib.crc32(key.encode()))
        seeds=rng.choice(ids,min(80,len(ids)),replace=False,p=m.area_faces[ids]/m.area_faces[ids].sum())
        for seed in seeds:
            target=rng.uniform(*WORK_AREA_FRACTION)*m.area
            mask,area=B.grow(m,int(seed),target,WORK_AREA_FRACTION[1]*m.area,~eligible,self.adj,self.weights)
            fraction=float(area/m.area)
            if WORK_AREA_FRACTION[0]<=fraction<=WORK_AREA_FRACTION[1] and B.components(m,mask)==1:
                return mask,dict(work_area_fraction=fraction,work_seed=int(seed),
                    work_components=1,work_normal_rays_clear=True,
                    minimum_work_vertex_z_m=float(world[m.faces[mask],2].min()),
                    minimum_work_outward_normal_z=float((m.face_normals@T[:3,:3].T)[mask,2].min()))
        raise ValueError(f'{key}: no connected 6-10% eligible working patch')


def export(name,mesh,initial,candidate,trial,poses,rule,checks,arm,
           generator='codes/setup/sequence.py',metadata=None,hand_xml=None):
    folder=ROOT/'objects'/name
    regions=getattr(trial,'regions',None) or WorkRegions(mesh)
    patches=[regions.choose(T,f'{name}/pose_{i}') for i,T in enumerate(poses,1)]
    names=[f'pose_{i}' for i in range(1,11)]
    record=dict(schema='cadgrasp_sequence_v1',object=name,coordinate_system='z_up_xy_floor',
        rest=dict(pose_id='rest',T_world_mesh=initial.tolist()),
        poses=[dict(index=i,pose_id=pose,T_world_mesh=T.tolist(),grounded=True,
                    robot_held=True) for i,(pose,T) in enumerate(zip(names,poses),1)],
        order=['rest']+names,transitions=[dict(source='rest' if i==0 else names[i-1],
             destination=pose,**{k:v for k,v in checks[i].items() if k!='pose_id'}) for i,pose in enumerate(names)],
        trajectory='trajectory.npz',sequence_continuous=True,object_reset_between_poses=False,
        transfer_mode='grasp, lift 25 mm, reorient, lower, hold on floor; same grasp throughout',
        grasp=dict(model=str(Path(hand_xml).resolve().relative_to(ROOT)) if hand_xml else 'codes/setup/assets/parallel_jaw.xml',T_world_hand=candidate['hand'].tolist(),
                   nominal_width_m=candidate['width'],opening_per_finger_m=trial.opening),pose_selection=rule,kuka_checks=arm,
        verification_scope='Free-object contact simulation, nominal exact-mesh ground poses, '
            'sampled robot IK/limits/speeds and scene collisions. Robot is kinematic; '
            'actuator torques, fixture installation and physical hardware are not certified.',
        object_floor_collision='Raw convex hull, exact support function for a plane',
        gripper_object_collision='Boundary-preserving tetrahedra' if name=='B' else 'Saved convex decomposition',
        controller='Perfect simulated object pose feedback; grasp transform re-estimated at each transition',
        generator=generator,generator_sha256=digest(ROOT/generator),
        source_hash_scope='Source files on disk at export; resumed checkpoint prefixes may have been generated by earlier revisions',
        mesh_sha256=digest(folder/'mesh.stl'))
    if metadata:
        record.update(metadata)
    if 'grasps' in record:
        for pose,grasp in zip(record['poses'],record['grasps']):
            pose['grasp_id']=grasp['grasp_id']
    with tempfile.TemporaryDirectory(prefix='cadgrasp-sequence-export-') as temp:
        stage=Path(temp)
        write(stage/'poses.json',record)
        for i,(T,(mask,area)) in enumerate(zip(poses,patches),1):
            pose=f'pose_{i}';out=stage/'tasks'/pose;out.mkdir(parents=True)
            vertices=trimesh.transform_points(mesh.vertices,T)
            ids=np.flatnonzero(vertices[:,2]<1e-8)
            assert len(ids)==1
            point=vertices[ids[0]].copy();point[2]=0.
            com=trimesh.transform_points(mesh.center_mass[None],T)[0]
            np.savez_compressed(out/'setup.npz',object=name,pose_id=pose,T_world_mesh=T,
                com_m=com,work_faces=mask,floor_contact_m=point,mesh_sha256=digest(folder/'mesh.stl'),
                poses_sha256=digest(stage/'poses.json'),K=.5,cone_half_deg=30.,tip=-1)
            report=dict(schema_version=1,object=name,pose_id=pose,label=f'Continuous sequence target {i}',
                target_pose_only=False,placement_trajectory_verified=True,
                placement_verification='Numerical contact simulation plus sampled robot checks; see poses.json',
                support_search_run=False,T_world_mesh=T.tolist(),floor_contact_m=point.tolist(),
                work_face_ids=np.flatnonzero(mask).tolist(),uniform_subdivision_rounds=regions.rounds,
                K=.5,cone_half_deg=30.,source_snapshot='setup.npz',source_snapshot_sha256=digest(out/'setup.npz'),
                checks=dict(**area,ground_min_z_m=float(vertices[:,2].min()),
                            floor_contact_raw_vertex_ids=ids.tolist(),com_world_m=com.tolist()),
                work_region_method='Seeded random area-weighted seed and random 6-10% target; '
                    'connected geodesic patch within visible upward faces and above floor',
                reachability_scope='Object normal rays; full load cone checked by baseline',
                generator='codes/setup/sequence_export.py')
            write(out/'setup.json',report)
        write(stage/'tasks.json',dict(schema='cadgrasp_tasks_v1',object=name,poses=names,
            rest='poses.json:rest',sequence='poses.json',trajectory='trajectory.npz',
            definition='Ten ordered grounded robot-held poses; consecutive transitions share one continuous simulation'))
        qpos,pos,quat=zip(*trial.history)
        extensions={}
        if getattr(trial,'lock_id',-1)>=0:
            assert len(trial.constraint_history)==len(qpos)
            extensions=dict(ideal_grasp_active=np.asarray(trial.constraint_history,dtype=bool),
                            ideal_grasp_eq_data=np.asarray(trial.constraint_data_history))
        np.savez_compressed(stage/'trajectory.npz',time_s=.033*(np.arange(len(qpos))+1),
            qpos=np.array(qpos),mocap_pos=np.array(pos),mocap_quat=np.array(quat),
            T_world_object=np.array(trial.object_history),target_transforms=np.array(poses),
            target_times_s=np.array([c['end_time_s'] for c in checks]),rest_transform=initial,
            robot_frame_indices=trial.arm_frames,robot_q=trial.arm_q,**extensions)
        # Replace only the task inputs after the complete new sequence is ready.
        if (folder/'tasks').exists():shutil.rmtree(folder/'tasks')
        shutil.move(str(stage/'tasks'),folder/'tasks')
        for file in ('poses.json','tasks.json','trajectory.npz'):
            shutil.copy2(stage/file,folder/file)
    from organize import publish_segments
    publish_segments(folder,invalidate_video=True)
    print(f'{name}: exported rest + 10 grounded targets and continuous trajectory',flush=True)
