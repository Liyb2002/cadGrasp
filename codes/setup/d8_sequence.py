"""Reproduce D8's continuous, motion-derived grounded target sequence.

A generic parallel-jaw grasp first tilts and seats the object. Nine subsequent
small yaw turns retain the same grip and floor contact. Every target orientation
comes from a measured settled state; the object is never reset between targets.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import json
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import grasp as G
import sequence as S
import kuka_transfer as K
from d2_derived import D2Trial
from sequence_export import WorkRegions, export, digest

REST_ROTATION = np.array([
    [0.9998706014386718, 3.0835030030479214e-05, -0.016086622637140606],
    [3.0835030030479214e-05, 0.9999926521665525, 0.003833361723758944],
    [0.016086622637140606, -0.003833361723758944, 0.9998632536052243]])
CANDIDATE_INDEX = 16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', action='store_true')
    args = parser.parse_args()
    folder = G.ROOT/'objects/D8'
    mesh = trimesh.load(folder/'mesh.stl', force='mesh')
    initial = S.seat(mesh, REST_ROTATION)
    model = G.build('D8', hand_xml=G.HERE/'assets/parallel_jaw.xml', floor_hull=True)
    regions = WorkRegions(mesh)
    candidate = next(c for i,c in enumerate(G.candidates(mesh, initial, 1,
        max_width=.15, depths=(.112,.118,.10))) if i==CANDIDATE_INDEX)
    assert G.geometry_check(model, initial, candidate)
    trial = D2Trial(model, mesh, initial, candidate, regions=regions)
    trial.warmup()
    axis = np.array([.23,1.,.13]); axis /= np.linalg.norm(axis)
    poses = []; checks = []; begin = 5.
    for step in range(40):
        if step==0:
            rotation = Rotation.from_rotvec(axis*np.deg2rad(6.)).as_matrix()@trial.start[:3,:3]
        else:
            rotation = Rotation.from_euler('z',3.,degrees=True).as_matrix()@G.transform(trial.data,trial.obj)[:3,:3]
        target = S.seat(mesh, rotation)
        pose,row,good = trial.move(target, lift=.025 if step==0 else 0.)
        separation = min((np.rad2deg(Rotation.from_matrix(pose[:3,:3]@p[:3,:3].T).magnitude())
            for p in poses), default=180.)
        print(json.dumps(dict(step=step,good=bool(good),separation=separation,**row)),flush=True)
        if good and separation>=.75:
            row.update(pose_id=f'pose_{len(poses)+1}',start_time_s=begin,end_time_s=trial.steps*.001)
            poses.append(pose); checks.append(row); begin=trial.steps*.001
            if len(poses)==10:
                break
    if len(poses)!=10:
        raise ValueError(f'Only {len(poses)} acceptable grounded holds')
    print('Starting full-pose KUKA and scene checks',flush=True)
    arm = K.check(trial.history)
    trial.arm_frames,trial.arm_q,arm['scene_checks'] = K.check_scene(model,trial.history,'D8')
    audit = mujoco.MjData(model); worst=0.
    for q,pos,quat in trial.history:
        audit.qpos[:]=q; audit.mocap_pos[:]=pos; audit.mocap_quat[:]=quat
        mujoco.mj_forward(model,audit)
        for contact in audit.contact:
            names=[model.geom(int(g)).name for g in (contact.geom1,contact.geom2)]
            if 'floor' in names and any(n.startswith('hand_geom_') for n in names):
                worst=min(worst,float(contact.dist))
    if worst<-.0002:
        raise ValueError(f'Gripper floor penetration: {-worst}')
    rule=dict(method='motion-derived',initial_tilt_axis_world=axis.tolist(),candidate_index=CANDIDATE_INDEX,
        first_command_tilt_deg=6.,subsequent_grounded_yaw_increment_deg=3.,
        minimum_pairwise_rotation_deg=.75,
        description='Measured stationary grounded orientations selected from one continuous contact simulation; only measured floor penetration is removed. Zero rotation error is definitional, not command tracking accuracy.',
        reproduction_command='python codes/setup/d8_sequence.py --export')
    print(json.dumps(dict(arm=arm,selected_targets=10)),flush=True)
    if args.export:
        export('D8',mesh,initial,candidate,trial,poses,rule,checks,arm,
            generator='codes/setup/d8_sequence.py',metadata=dict(
                transfer_mode='One pickup and initial 25 mm lift/tilt/lower, then grounded world-z yaw turns with the same grasp; no object resets. Unselected exploration states, if any, remain in the trajectory.',
                verification_scope='Motion-derived contact simulation, sampled KUKA IK/limits/speeds and scene collisions. Robot kinematic; hardware and actuator torque not certified.',
                generator_dependencies={f'codes/setup/{name}':digest(G.HERE/name)
                    for name in ('d1_sequence.py','d2_derived.py')},
                verification=dict(gripper_floor_maximum_penetration_m=-worst)))
        from verify_sequences import verify
        verify(folder)


if __name__=='__main__':
    main()
