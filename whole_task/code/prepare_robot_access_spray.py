"""Native Panda small-angle spraying path for the existing robot_access scene.

This is a checked kinematic illustration with ray-clipped spray particles,
not a new shared-fixture solution or a paint/dynamics certificate.
"""
from pathlib import Path
import hashlib
import json
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'slides/idea/data/robot_access'
ASSETS=ROOT/'slides/Co-optimize/idea/assets'
OUTPUT=ROOT/'whole_task/assets/robot_access_spray'
sys.path.insert(0,str(ASSETS))
from franka import Franka


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    source=json.loads((SOURCE/'scene.json').read_text())
    parent=ROOT/'robots_config/01_franka_panda.json'
    config=json.loads(parent.read_text())
    H0=np.array(source['tool_hand_transform'])
    axis=H0[:3,2]
    target=np.array(source['front_contact'])
    config['task_constraint']['working_direction_world']=axis.tolist()
    config['task_constraint']['cone_half_angle_deg']=10
    config['task_constraint']['scenario_zh']='robot_access 固定正面工位的小角度喷涂示意'
    config['task_constraint']['scenario_en']='Small-angle spraying at the existing front workstation'
    config['task_constraint']['estimate_rationale_zh']='保留 Panda 本体参数；工位方向改为原场景的正面方向，允许半角收紧至 10 度。动画只展示其中约 3 度的实际小幅路径，不声称整个锥均可达。'
    config['parent_config']='robots_config/01_franka_panda.json'
    config['parent_config_sha256']=sha(parent)
    config['figure']=None
    config['station_source']='slides/idea/data/robot_access/scene.json'
    (OUTPUT/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    arm=Franka()
    arm.model.body_pos[arm.model.body('link0').id]=0
    visual=mujoco.MjModel.from_xml_path(str(ASSETS/'franka_emika_panda/panda.xml'))
    data=mujoco.MjData(visual)
    obj=trimesh.load(SOURCE/'object.obj',force='mesh',process=False)
    fixture=trimesh.load(SOURCE/'fixture.obj',force='mesh',process=False)
    combined=trimesh.util.concatenate([obj,fixture])
    front=set(source['patches']['17']['face_ids'])
    rng=np.random.default_rng(20261016)
    count=100
    radius=np.sqrt(rng.random(count))*np.tan(np.radians(4))
    azimuth=rng.random(count)*2*np.pi
    offsets=np.c_[radius*np.cos(azimuth),radius*np.sin(azimuth),np.ones(count)]
    offsets/=np.linalg.norm(offsets,axis=1,keepdims=True)
    phases=rng.random(count)
    frames=[]; joints=[]; poses=[]; tips=[]; endpoints=[]; masks=[]
    body_poses=[]; errors=[]; incidence=[]; deviations=[]; center_hits=[]
    previous=np.array(source['robot_joints'])
    fps=24; total=192; tool_length=.103; standoff=.052
    for frame in range(total):
        phase=frame/(total-1)
        angles=np.radians([2.5*np.sin(2*np.pi*phase),-.75*(1-np.cos(2*np.pi*phase))])
        H=H0.copy(); H[:3,:3]=H0[:3,:3]@Rotation.from_euler('xy',angles).as_matrix()
        H[:3,3]=target-H[:3,2]*(tool_length+standoff)
        q=arm.ik(H,start=previous,multistart=False)
        if q is None: raise RuntimeError(f'IK failed at frame {frame}')
        actual=arm.fk(q).copy(); previous=q
        nozzle=actual[:3,3]+actual[:3,2]*tool_length
        hits,_,faces=combined.ray.intersects_location([nozzle],[actual[:3,2]],multiple_hits=True)
        distances=(hits-nozzle)@actual[:3,2]
        positive=np.flatnonzero(distances>1e-6)
        first=positive[np.argmin(distances[positive])]
        if int(faces[first]) not in front: raise RuntimeError('Center spray ray missed the declared front area')
        errors.append(np.linalg.norm(hits[first]-target))
        center_hits.append(hits[first])
        incidence.append(np.degrees(np.arccos(np.clip(-obj.face_normals[faces[first]]@actual[:3,2],-1,1))))
        deviations.append(np.degrees(np.arccos(np.clip(axis@actual[:3,2],-1,1))))
        directions=offsets@actual[:3,:3].T
        hit,which,face=combined.ray.intersects_location(np.repeat(nozzle[None],count,axis=0),directions,multiple_hits=True)
        distance=np.einsum('ij,ij->i',hit-nozzle,directions[which])
        clip=np.full(count,np.inf); terminal=np.zeros((count,3)); valid=np.zeros(count,bool)
        for p,ray,face_id,dist in zip(hit,which,face,distance):
            if 1e-6<dist<clip[ray]:
                clip[ray]=dist; terminal[ray]=p; valid[ray]=int(face_id) in front
        if not valid.all(): raise RuntimeError('A spray ray left the selected front work region')
        data.qpos[:7]=q; mujoco.mj_forward(visual,data)
        transforms=[]
        for i in range(8):
            body=visual.body(f'link{i}').id
            T=np.eye(4);T[:3,:3]=data.xmat[body].reshape(3,3);T[:3,3]=data.xpos[body]
            transforms.append(T)
        body_poses.append(transforms)
        joints.append(q); poses.append(actual); tips.append(nozzle); endpoints.append(terminal); masks.append(valid)
    q=np.array(joints); poses=np.array(poses)
    assert (q>arm.lower).all() and (q<arm.upper).all()
    speeds=np.abs(np.diff(q,axis=0))*fps
    limits=np.array([2.175]*4+[2.61]*3)
    assert (speeds<=limits).all()
    assert max(deviations)<10 and max(incidence)<10 and max(errors)<2e-5
    np.savez_compressed(OUTPUT/'trajectory.npz',joint_q=q,hand_transforms=poses,
        link_transforms=np.array(body_poses),nozzles=np.array(tips),spray_endpoints=np.array(endpoints),
        spray_valid=np.array(masks),particle_phases=phases,particle_offsets=offsets,center_hits=np.array(center_hits))
    # Export visual parts in each link's LOCAL frame. No manipulation of robot geometry.
    data.qpos[:7]=q[0];mujoco.mj_forward(visual,data)
    groups={}
    for g in range(visual.ngeom):
        name=visual.body(visual.geom_bodyid[g]).name
        if not name.startswith('link') or visual.geom_group[g]!=2: continue
        mid=visual.geom_dataid[g]
        lettering=visual.mesh(mid).name=='link5_1'
        va,fa=visual.mesh_vertadr[mid],visual.mesh_faceadr[mid]
        vertices=visual.mesh_vert[va:va+visual.mesh_vertnum[mid]].copy()
        faces=visual.mesh_face[fa:fa+visual.mesh_facenum[mid]].copy()
        vertices=vertices@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]
        body=visual.geom_bodyid[g]
        vertices=(vertices-data.xpos[body])@data.xmat[body].reshape(3,3)
        if lettering:
            # Removing the raised letters alone leaves letter-shaped gaps in
            # the adjacent housing. Cover that label with a blank dark plate.
            center=vertices.mean(axis=0)
            _,_,basis=np.linalg.svd(vertices-center,full_matrices=False)
            local=(vertices-center)@basis.T
            low,high=local.min(axis=0),local.max(axis=0)
            low[:2]-=.002;high[:2]+=.002
            low[2]-=.0012;high[2]+=.0012
            corners=np.array([[x,y,z] for x in [low[0],high[0]] for y in [low[1],high[1]] for z in [low[2],high[2]]])
            cover=corners@basis+center
            cover_faces=np.array([[0,4,6],[0,6,2],[1,3,7],[1,7,5],[0,1,5],[0,5,4],
                                  [2,6,7],[2,7,3],[0,2,3],[0,3,1],[4,5,7],[4,7,6]])
            groups.setdefault((int(name[4:]),'black'),[]).append((cover,cover_faces))
            continue
        white=np.mean(visual.mat_rgba[visual.geom_matid[g],:3])>.6
        groups.setdefault((int(name[4:]),'white' if white else 'black'),[]).append((vertices,faces))
    arrays={}; parts=[]
    for (body,color),meshes in groups.items():
        name=f'link{body}_{color}';vertices=[];faces=[];offset=0
        for v,f in meshes:
            vertices.append(v);faces.append(f+offset);offset+=len(v)
        arrays[name+'_v']=np.concatenate(vertices);arrays[name+'_f']=np.concatenate(faces)
        parts.append(dict(name=name,body=body,color=color))
    np.savez_compressed(OUTPUT/'robot_meshes.npz',**arrays)
    metadata=dict(fps=fps,frames=total,duration_s=total/fps,tool_length_m=tool_length,standoff_m=standoff,
        nominal_axis_world=axis.tolist(),target_world=target.tolist(),robot_parts=parts,
        source_scene='slides/idea/data/robot_access/scene.json',source_scene_sha256=sha(SOURCE/'scene.json'),
        parent_config_sha256=sha(parent),config_sha256=sha(OUTPUT/'config.json'),
        source_mesh_hashes={n:sha(SOURCE/n) for n in ['object.obj','fixture.obj','front_patch.obj','back_patch.obj']},
        verification=dict(joint_limits_pass=True,joint_speed_limits_pass=True,
            max_joint_step_deg=float(np.degrees(np.max(np.abs(np.diff(q,axis=0))))),
            max_joint_speed_rad_s=float(speeds.max()),max_tool_deviation_deg=float(max(deviations)),
            max_surface_incidence_deg=float(max(incidence)),max_center_ray_target_error_m=float(max(errors)),
            all_spray_rays_first_hit_front_region=True,ray_clipping_includes_fixture=True,
            checked_spray_rays=total*count,physics_or_full_scene_collision_certificate=False),
        scope='Small saved-Panda kinematic path and geometric spray visualization; not a new pose/fixture solve or paint process certification.',
        lettering_mesh_omitted='link5_1',lettering_replaced_with_blank_dark_plate=True,
        no_text_or_numbers_in_video=True)
    (OUTPUT/'motion.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata['verification'],indent=2))


if __name__=='__main__': main()
