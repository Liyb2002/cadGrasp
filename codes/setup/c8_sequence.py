"""Reproduce C8's motion-derived, continuously reached grounded target sequence.

No object reset occurs after initialization. Targets are selected AFTER physics
integration; this does not certify accurate tracking of a predefined pose menu.
The generic parallel jaw, contact parameters and KUKA base are unchanged.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'codes/setup'))
import json,hashlib
import mujoco,numpy as np,trimesh
from scipy.spatial.transform import Rotation
import sequence as S
import grasp as G
from sequence_export import WorkRegions,export,digest,write


def main():
    folder=ROOT/'objects/C8'
    mesh=trimesh.load(folder/'mesh.stl',force='mesh')
    rest=dict(T_world_mesh=[[0.45810759956023545, -0.3959533502266062, -0.7958381567061771, 0.012068627335016683], [-0.3959533502266062, 0.710682313631928, -0.581508033938566, -0.014575593885378798], [0.7958381567061771, 0.581508033938566, 0.16878991319216347, 0.006205255668451877], [0.0, 0.0, 0.0, 1.0]])
    initial=S.seat(mesh,np.asarray(rest['T_world_mesh'])[:3,:3])
    model=G.build('C8',hand_xml=G.HERE/'assets/parallel_jaw.xml',floor_hull=True)
    candidate=next(c for i,c in enumerate(G.candidates(mesh,initial,60,max_width=.15,
        depths=(.112,.118,.1))) if i==507)
    assert G.geometry_check(model,initial,candidate)
    regions=WorkRegions(mesh)
    trial=S.Trial(model,mesh,initial,candidate,regions=regions)
    trial.warmup()
    intended,proposal_rule=trial.proposals()
    initial_tracking_report=None
    try:
        trial.sequence(intended[:1])
    except ValueError as error:
        # The realized state is not labelled a successful prescribed placement.
        initial_tracking_report=str(error)
    h=G.transform(trial.data,trial.hand)
    for _ in range(2000):trial.step(h)
    poses=[];checks=[];snapshots=[];previous_end=0.
    for stage in range(50):
        hprev=h.copy();window=[]
        for j in range(6000):
            rotation=Rotation.from_euler('z',3*G.smooth(j/2000),degrees=True).as_matrix()
            h[:3,:3]=rotation@hprev[:3,:3]
            h[:2,3]=(rotation@hprev[:3,3])[:2]
            actual=G.transform(trial.data,trial.obj)
            gap=float((trial.floor_vertices@actual[2,:3]).min()+actual[2,3])
            h[2,3]+=np.clip(.001*2*(-.0000005-gap),-.000002,.000002)
            trial.step(h)
            if j>=5000 and j%20==0:window.append(G.transform(trial.data,trial.obj))
        mujoco.mj_forward(model,trial.data)
        actual=G.transform(trial.data,trial.obj)
        vertices=trimesh.transform_points(mesh.vertices,actual);gap=float(vertices[:,2].min())
        fingers,force=S.contacts(model,trial.data)
        angle_drift=max(np.rad2deg(Rotation.from_matrix(T[:3,:3]@actual[:3,:3].T).magnitude()) for T in window)
        position_drift=max(np.linalg.norm(T[:3,3]-actual[:3,3]) for T in window)
        point=vertices[vertices[:,2].argmin()]
        com=actual[:3,:3]@mesh.center_mass+actual[:3,3]
        margin=float(np.linalg.norm(com[:2]-point[:2]))
        adjacent=(np.rad2deg(Rotation.from_matrix(actual[:3,:3]@poses[-1][:3,:3].T).magnitude()) if poses else 180.)
        T=actual.copy();T[2,3]-=gap
        ids=np.flatnonzero(trimesh.transform_points(mesh.vertices,T)[:,2]<1e-8)
        usable=(fingers==2 and force>.001 and -.001<gap<.0003 and angle_drift<.25
            and position_drift<.0003 and margin>.002 and adjacent>1.5 and len(ids)==1
            and regions.feasible(T))
        print(json.dumps(dict(stage=stage,selected=bool(usable),count=len(poses),gap=gap,force=force,
            drift_deg=angle_drift,drift_m=position_drift,adjacent_deg=adjacent,com_offset_m=margin)),flush=True)
        if not usable:continue
        index=len(poses)+1;poses.append(T)
        checks.append(dict(pose_id=f'pose_{index}',start_time_s=previous_end,end_time_s=trial.steps*.001,
            target_position_error_m=abs(gap),target_rotation_error_deg=0.,raw_mesh_floor_gap_m=gap,
            object_floor_normal_force_N=force,bilateral_grip=True,selection_stage=stage,
            target_is_motion_derived=True,hold_observation_s=1.,hold_max_rotation_drift_deg=angle_drift,
            hold_max_translation_drift_m=float(position_drift),adjacent_rotation_deg=float(adjacent),
            com_horizontal_offset_from_lowest_vertex_m=margin))
        snapshots.append((trial.data.qpos.copy(),trial.data.qvel.copy(),trial.data.mocap_pos.copy(),trial.data.mocap_quat.copy()))
        previous_end=trial.steps*.001
        if len(poses)==10:break
    if len(poses)!=10:raise RuntimeError('Fewer than ten valid realized holds')
    arm=S.K.check(trial.history)
    trial.arm_frames,trial.arm_q,arm['scene_checks']=S.K.check_scene(model,trial.history,'C8')
    audit=mujoco.MjData(model);worst_gripper_floor=0.
    for qpos,pos,quat in trial.history:
        audit.qpos[:]=qpos;audit.mocap_pos[:]=pos;audit.mocap_quat[:]=quat
        mujoco.mj_forward(model,audit)
        for contact in audit.contact:
            names=[model.geom(int(g)).name for g in (contact.geom1,contact.geom2)]
            if 'floor' in names and any(n.startswith('hand_geom_') for n in names):
                worst_gripper_floor=min(worst_gripper_floor,float(contact.dist))
    assert worst_gripper_floor>=-.0002
    arm['gripper_floor_checks']=dict(samples=len(trial.history),
        maximum_penetration_m=-worst_gripper_floor,collision_tolerance_m=.0002)
    # Independent gravity-only counterfactuals: remove gripper collision, never
    # feed these branches back to the uninterrupted exported sequence.
    released=G.build('C8',hand_xml=G.HERE/'assets/parallel_jaw.xml',floor_hull=True)
    for g in range(released.ngeom):
        if released.geom(g).name.startswith('hand_geom_'):
            released.geom_contype[g]=0;released.geom_conaffinity[g]=0
    for T,row,snapshot in zip(poses,checks,snapshots):
        data=mujoco.MjData(released)
        data.qpos[:],data.qvel[:],data.mocap_pos[:],data.mocap_quat[:]=snapshot
        mujoco.mj_forward(released,data)
        before=G.transform(data,released.body('object').id)
        for _ in range(1000):mujoco.mj_step(released,data)
        after=G.transform(data,released.body('object').id)
        drift=float(np.rad2deg(Rotation.from_matrix(after[:3,:3]@before[:3,:3].T).magnitude()))
        row['gripper_removed_1s_rotation_drift_deg']=drift
        if drift<2:raise ValueError(f'Target does not demonstrate support need: {drift}')
    rule=dict(method='motion-derived: select realized quasi-static grounded holds after continuous dynamics',
        commanded_stage_yaw_deg=3.,minimum_adjacent_measured_rotation_deg=1.5,
        maximum_hold_rotation_drift_deg=.25,hold_observation_s=1.,
        rest_placement=0,grasp_candidate=507,initial_proposal=proposal_rule,
        initial_prescribed_pose_tracking_failure=initial_tracking_report,
        accuracy_interpretation='Targets use realized rotation and XY translation, with only a micron-scale vertical seating correction. Zero target rotation error is definitional, not a tracking accuracy claim.',
        support_need_test='Independent 1 s gravity-only branches disable gripper collision; object drift must exceed 2 degrees. These branches are not in the exported trajectory.')
    export('C8',mesh,initial,candidate,trial,poses,rule,checks,arm)
    # Update provenance and transfer description, then refresh dependent hashes.
    record=json.loads((folder/'poses.json').read_text())
    record.update(generator='codes/setup/c8_sequence.py',generator_sha256=digest(__file__),
        transfer_mode='Single continuous grasp; initial lift and settling, followed by 3 degree yaw increments with vertical ground-contact feedback and 4 s dwell per stage',
        controller='Measured object-height feedback drives gripper vertical position; slow commanded world-Z yaw. Poses are selected from realized holds.',
        verification_scope='Free-object contact simulation and observed hold stability; motion-derived targets, not predefined pose tracking. Sampled KUKA full-pose IK, limits/speeds and scene collisions. Robot kinematic; hardware, actuator torques and fixture installation not certified.')
    write(folder/'poses.json',record)
    for index in range(1,11):
        p=folder/'tasks'/f'pose_{index}'
        with np.load(p/'setup.npz') as z:values={key:z[key] for key in z.files}
        values['poses_sha256']=digest(folder/'poses.json')
        np.savez_compressed(p/'setup.npz',**values)
        report=json.loads((p/'setup.json').read_text())
        report.update(source_snapshot_sha256=digest(p/'setup.npz'),placement_verification='Motion-derived realized hold with micron-scale seating correction; continuous free-object simulation and sampled robot checks. See poses.json.')
        write(p/'setup.json',report)
    from verify_sequences import verify
    assert verify(folder)==10
    print(json.dumps(dict(passed=True,object='C8',targets=10,duration_s=trial.steps*.001,
        max_hold_drift_deg=max(c['hold_max_rotation_drift_deg'] for c in checks),
        minimum_release_drift_deg=min(c['gripper_removed_1s_rotation_drift_deg'] for c in checks),kuka=arm)),flush=True)

if __name__=='__main__':main()
