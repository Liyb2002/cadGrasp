"""Full-pose KUKA IK for a sampled gripper trajectory; no arm dynamics claim."""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'simulation'))
from kuka import Arm


def add_render_arm(root, collisions=False):
    """Add the real KUKA visuals to a replay scene; these are not arm dynamics."""
    import xml.etree.ElementTree as ET
    from kuka import add_robots
    add_robots(root)
    world = root.find('worldbody')
    world.remove(world.find("body[@name='robot_B_link_0']"))
    robot = world.find("body[@name='robot_A_link_0']")
    robot.set('pos','-.45 -.35 0')
    wrist = robot.find(".//body[@name='robot_A_link_7']")
    for geom in list(wrist.findall('geom')):
        if geom.get('name') in ('robot_A_palm','robot_A_tool_stem'):
            wrist.remove(geom)
    ET.SubElement(wrist,'geom',type='cylinder',pos='0 0 .0175',size='.021 .0175',
                  rgba='.28 .30 .32 1',contype='0',conaffinity='0')
    ET.SubElement(root.find('asset'),'texture',name='replay_sky',type='skybox',
                  builtin='gradient',rgb1='.95 .97 .98',rgb2='.95 .97 .98',width='64',height='64')
    root.find('visual/headlight').attrib.update(ambient='.55 .55 .55',diffuse='.65 .65 .65')
    if collisions:
        from kuka import ASSETS,description
        links,_,_=description()
        for index,link in enumerate(links):
            ET.SubElement(root.find('asset'),'mesh',name=f'arm_collision_{index}',
                          file=str(ASSETS/f'meshes/collision/link_{index}.stl'))
            body=robot if index==0 else robot.find(f".//body[@name='robot_A_link_{index}']")
            ET.SubElement(body,'geom',name=f'arm_collision_{index}',type='mesh',
                          mesh=f'arm_collision_{index}',pos=link.find('collision/origin').get('xyz'),
                          group='3',contype='16',conaffinity='23')
        contact=root.find('contact')
        for index in range(7):
            ET.SubElement(contact,'exclude',body1=f'robot_A_link_{index}',
                          body2=f'robot_A_link_{index+1}')
        ET.SubElement(contact,'exclude',body1='robot_A_link_7',body2='hand')


def repair_scene_configuration(arm,target,seed,model,data,indices,previous=None,dt=None):
    """Adjust the redundant arm posture while preserving the tool pose.

    Only the robot's seven kinematic joints are optimized. The recorded hand
    and free-object state remain fixed throughout these collision queries.
    """
    import mujoco
    pairs=set()
    lower,upper=arm.lower.copy(),arm.upper.copy()
    if previous is not None and dt is not None:
        lower=np.maximum(lower,previous-arm.velocity*dt)
        upper=np.minimum(upper,previous+arm.velocity*dt)
    q=np.clip(seed,lower+1e-9,upper-1e-9)
    for _ in range(3):
        data.qpos[indices]=q;mujoco.mj_forward(model,data)
        for contact in data.contact:
            ids=(int(contact.geom1),int(contact.geom2))
            if contact.dist<-.0002 and any(model.geom(g).name.startswith('arm_collision_') for g in ids):
                pairs.add(tuple(sorted(ids)))
        fixed_pairs=sorted(pairs)
        def residual(value):
            position,rotation,transforms=flange(arm,value)
            data.qpos[indices]=value;mujoco.mj_forward(model,data)
            distances=[mujoco.mj_geomDistance(model,data,a,b,.01,None) for a,b in fixed_pairs]
            return np.r_[position-target[:3,3],
                .2*Rotation.from_matrix(rotation@target[:3,:3].T).as_rotvec(),
                5*max(0.,.002-floor_clearance(arm,transforms)),
                10*np.maximum(0.,.001-np.asarray(distances)),1e-5*(value-seed)]
        fit=least_squares(residual,q,bounds=(lower,upper),max_nfev=160,
                          ftol=1e-10,xtol=1e-10,gtol=1e-10)
        q=fit.x
        position,rotation,transforms=flange(arm,q)
        data.qpos[indices]=q;mujoco.mj_forward(model,data)
        colliding=any(c.dist<-.0002 and
            any(model.geom(int(g)).name.startswith('arm_collision_') for g in (c.geom1,c.geom2))
            for c in data.contact)
        if (not colliding and np.linalg.norm(position-target[:3,3])<1e-4
                and Rotation.from_matrix(rotation@target[:3,:3].T).magnitude()<1e-3
                and floor_clearance(arm,transforms)>=-.0001):
            return q
    remaining=[(model.geom(int(c.geom1)).name,model.geom(int(c.geom2)).name,float(c.dist))
        for c in data.contact if c.dist<-.0002 and
        any(model.geom(int(g)).name.startswith('arm_collision_') for g in (c.geom1,c.geom2))]
    raise ValueError('Collision-aware KUKA IK repair failed: '
        f'position={np.linalg.norm(position-target[:3,3]):.6g} m, '
        f'rotation={Rotation.from_matrix(rotation@target[:3,:3].T).magnitude():.6g} rad, '
        f'contacts={remaining}')


def check_scene(source,history,name,stride=3,initial_q=None,hand_xml=None,robot_track=None):
    """Sample robot/scene/self collisions in addition to six-dimensional IK."""
    import mujoco
    import grasp as G
    model=G.build(name,hand_xml=hand_xml or G.HERE/'assets/parallel_jaw.xml',render_arm=True,
                  robot_collisions=True,floor_hull=True)
    data=mujoco.MjData(model)
    arm=Arm('A');arm.base[:]=[-.45,-.35,0.]
    indices=[model.joint(f'robot_A_joint_{i}').qposadr[0] for i in range(1,8)]
    qs=[];frames=[];previous=initial_q;worst=0.;repairs=0;speed_repairs=0
    for index in sorted(set(range(0,len(history),stride))|{len(history)-1}):
        qpos,pos,quat=history[index]
        data.qpos[:source.nq]=qpos;data.mocap_pos[:]=pos;data.mocap_quat[:]=quat
        mujoco.mj_forward(model,data)
        target=G.transform(data,model.body('hand').id)
        seed=previous
        if robot_track is not None and index<=robot_track[0][-1]:
            track_frames,track_q=robot_track
            solve_seed=np.array([np.interp(index,track_frames,track_q[:,j]) for j in range(7)])
        else:
            solve_seed=previous
        try:
            previous=(solve_seed.copy() if robot_track is not None and
                      configuration_matches(arm,target,solve_seed) else solve(arm,target,solve_seed)[0])
        except ValueError as error:
            raise ValueError(f'KUKA IK failed at local frame {index}, hand position {target[:3,3].tolist()}') from error
        data.qpos[indices]=previous;mujoco.mj_forward(model,data)
        colliding=any(c.dist<-.0002 and
               any(model.geom(int(g)).name.startswith('arm_collision_') for g in (c.geom1,c.geom2))
               for c in data.contact)
        dt=None if not frames else (index-frames[-1])*.033
        too_fast=dt is not None and np.any(np.abs(previous-seed)>arm.velocity*dt)
        if colliding or too_fast:
            previous=repair_scene_configuration(arm,target,previous,model,data,indices,
                previous=seed,dt=dt)
            repairs+=int(colliding);speed_repairs+=int(too_fast)
            data.qpos[indices]=previous;mujoco.mj_forward(model,data)
        for contact in data.contact:
            names=[model.geom(int(g)).name for g in (contact.geom1,contact.geom2)]
            if not any(n.startswith('arm_collision_') for n in names):continue
            worst=min(worst,float(contact.dist))
            if contact.dist < -.0002:
                raise ValueError(f'Robot scene collision: {names}, penetration={-contact.dist:.6g} m')
        qs.append(previous);frames.append(index)
    ratio=float((np.abs(np.diff(qs,axis=0))/ (np.diff(frames)[:,None]*.033)/arm.velocity).max())
    if ratio>1:raise ValueError(f'Robot velocity limit exceeded: {ratio}')
    return np.array(frames),np.array(qs),dict(samples=len(frames),max_speed_limit_ratio=ratio,
        collision_repaired_samples=repairs,
        speed_repaired_samples=speed_repairs,
        maximum_penetration_m=-worst,collision_tolerance_m=.0002,
        collision_model='Convex hull of each URDF collision mesh; sampled, adjacent links excluded')


def reconstruct_track(source,history,name,checks,events,hand_xml=None):
    """Reconstruct old accepted piecewise IK paths as seeds for dense checks.

    Historical checkpoints retained only each final joint configuration.
    Pickup/target event boundaries reproduce their original stride-nine IK
    queries. This does not replace the subsequent densely sampled check.
    """
    pickups=[event for event in events if event['event']=='new_grasp']
    if len(pickups)!=len(checks):
        raise ValueError('Cannot reconstruct robot track: pickup/target counts differ')
    frames=[];configurations=[];previous=None
    for check,pickup in zip(checks,pickups):
        start=max(0,int(check['start_time_s']/.033)-1)
        middle=min(len(history)-1,int(pickup['end_time_s']/.033)-1)
        end=min(len(history)-1,int(check['end_time_s']/.033)-1)
        for begin,finish in ((start,middle),(middle,end)):
            local,q,_=check_scene(source,history[begin:finish+1],name,
                stride=int(check.get('robot_check_stride',9)),initial_q=previous,hand_xml=hand_xml)
            previous=q[-1]
            for frame,value in zip(local+begin,q):
                if frames and frame==frames[-1]:
                    continue
                frames.append(int(frame));configurations.append(value)
    return np.asarray(frames),np.asarray(configurations)


def render(source, history, output, name, tool, robot_track=None,hand_xml=None):
    """Replay integrated free-object motion in one unobstructed camera view.

    Only an MP4 is written. Arm motion is kinematic; the grasped object's saved
    dynamics are not changed to match the robot or the proposed seated poses.
    """
    import json
    import shutil
    import tempfile
    import mujoco
    import grasp as G
    sys.path.insert(0,str(G.ROOT/'slides/baseline_algo'))
    from step5_connect_support.video import mp4_writer
    model = G.build(name,hand_xml=hand_xml or (G.HERE/'assets/parallel_jaw.xml' if tool=='parallel' else None),
                    render_arm=True,robot_collisions=True,floor_hull=True)
    np.testing.assert_array_equal(source.jnt_qposadr,model.jnt_qposadr[:source.njnt])
    data = mujoco.MjData(model)
    arm = Arm('A'); arm.base[:] = [-.45,-.35,0.]
    robot_joints = [model.joint(f'robot_A_joint_{i}').qposadr[0] for i in range(1,8)]
    configurations, previous = [], None
    for frame,(qpos,pos,quat) in enumerate(history):
        data.qpos[:source.nq] = qpos
        data.mocap_pos[:] = pos; data.mocap_quat[:] = quat
        mujoco.mj_forward(model,data)
        target = G.transform(data,model.body('hand').id)
        if robot_track is not None:
            frames,qs=robot_track
            seed=np.array([np.interp(frame,frames,qs[:,j]) for j in range(7)])
        else:
            seed=previous
        preceding=previous
        previous=(seed.copy() if robot_track is not None and
                  configuration_matches(arm,target,seed) else solve(arm,target,seed)[0])
        data.qpos[robot_joints]=previous;mujoco.mj_forward(model,data)
        colliding=any(c.dist<-.0002 and
               any(model.geom(int(g)).name.startswith('arm_collision_') for g in (c.geom1,c.geom2))
               for c in data.contact)
        too_fast=preceding is not None and np.any(np.abs(previous-preceding)>arm.velocity*.033)
        if colliding or too_fast:
            previous=repair_scene_configuration(arm,target,previous,model,data,robot_joints,
                previous=preceding,dt=.033 if preceding is not None else None)
        configurations.append(previous)
    ratio = float((np.abs(np.diff(configurations,axis=0))/.033/arm.velocity).max())
    if ratio > 1.:
        raise ValueError(f'Rendered KUKA trajectory exceeds joint speed limit: {ratio}')
    print(json.dumps(dict(rendered_kuka_frames=len(history),joint_speed_limit_ratio=ratio)),flush=True)
    option = mujoco.MjvOption(); option.geomgroup[3] = 0
    wide = mujoco.MjvCamera(); wide.lookat[:] = [-.17,-.11,.28]
    wide.distance,wide.azimuth,wide.elevation = 1.20,125,-24
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='cadgrasp-kuka-video-') as temp:
        path = Path(temp)/output.name
        with mujoco.Renderer(model,height=800,width=1280) as renderer, mp4_writer(path,1000/33) as writer:
            for frame,((qpos,pos,quat),q) in enumerate(zip(history,configurations)):
                data.qpos[:source.nq] = qpos; data.qpos[robot_joints] = q
                data.mocap_pos[:] = pos; data.mocap_quat[:] = quat
                mujoco.mj_forward(model,data)
                renderer.update_scene(data,camera=wide,scene_option=option)
                writer.append_data(renderer.render())
                if frame % 150 == 0:
                    print(f'rendered {frame}/{len(history)}',flush=True)
        shutil.move(str(path),output)
    return np.asarray(configurations)


def flange(arm, q):
    _, rotation, transforms = arm.forward(q, geometry=True)
    # Upstream link_7 -> flange, without the old pushing palm's 100 mm stem.
    return transforms[-1][1] + .035*rotation[:,2], rotation, transforms


def floor_clearance(arm, transforms):
    return min((vertices@rotation.T+point)[:,2].min()
               for vertices,(rotation,point) in zip(arm.collisions[1:],transforms[1:]))


def configuration_matches(arm,target,q):
    """Reuse a recorded seed only after checking its actual forward pose."""
    if q is None or np.any(q<arm.lower) or np.any(q>arm.upper):return False
    position,rotation,transforms=flange(arm,q)
    return (np.linalg.norm(position-target[:3,3])<1e-5 and
            Rotation.from_matrix(rotation@target[:3,:3].T).magnitude()<1e-4 and
            floor_clearance(arm,transforms)>=-.0001)


def solve(arm, target, previous=None):
    seeds = ([previous] if previous is not None else
             list(np.random.default_rng(20260918).uniform(arm.lower*.75,arm.upper*.75,(24,7))))
    for seed in seeds:
        def residual(q):
            position, rotation, transforms = flange(arm,q)
            return np.r_[position-target[:3,3],
                .2*Rotation.from_matrix(rotation@target[:3,:3].T).as_rotvec(),
                5*max(0.,.002-floor_clearance(arm,transforms)),1e-6*(q-seed)]
        fit = least_squares(residual,seed,bounds=(arm.lower,arm.upper),max_nfev=200,
                            ftol=1e-10,xtol=1e-10,gtol=1e-10)
        position, rotation, transforms = flange(arm,fit.x)
        error = np.linalg.norm(position-target[:3,3])
        angle = Rotation.from_matrix(rotation@target[:3,:3].T).magnitude()
        clearance = floor_clearance(arm,transforms)
        if error < 1e-4 and angle < 1e-3 and clearance >= -.0001:
            return fit.x, (error,angle,clearance)
    raise ValueError('Full-pose KUKA IK or sampled arm/floor clearance failed')


def check(history, stride=3, initial_q=None):
    arm = Arm('A')
    # Demo cell placement is free to choose; robot geometry is never scaled.
    arm.base[:] = [-.45,-.35,0.]
    previous = initial_q
    errors, configurations, times = [], [], []
    for index in sorted(set(range(0,len(history),stride)) | {len(history)-1}):
        _, pos, quat = history[index]
        target = np.eye(4)
        target[:3,3] = pos[0]
        target[:3,:3] = Rotation.from_quat(quat[0][[1,2,3,0]]).as_matrix()
        previous, error = solve(arm,target,previous)
        errors.append(error); configurations.append(previous); times.append(index*.033)
    errors = np.asarray(errors)
    speed = np.abs(np.diff(configurations,axis=0))/np.diff(times)[:,None]
    ratio = float((speed/arm.velocity).max())
    if ratio > 1.:
        raise ValueError(f'Sampled KUKA joint velocity limit exceeded: ratio={ratio}')
    return dict(samples=len(errors), base_world_m=arm.base.tolist(),
        max_position_error_m=float(errors[:,0].max()),
        max_rotation_error_deg=float(np.degrees(errors[:,1].max())),
        min_arm_floor_clearance_m=float(errors[:,2].min()),
        max_joint_velocity_limit_ratio=ratio,
        scope='Sampled full-pose IK, joint limits/speeds and arm-floor clearance only; '
              'self/object/fixture collisions, mounting hardware and arm dynamics unchecked')
