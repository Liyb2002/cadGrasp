"""Reproduce A4's continuous sequence with a wide initial opening and fixed rest.

Yaw rotates the saved stable preparation and antipodal grasp together. Every
trial is then integrated afresh; the object is never reset between targets.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import json
import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
import grasp as G
import kuka_transfer as K
import sequence as S
from sequence_export import WorkRegions, export

REST_ROTATION = np.array([[0.9997580046794707, -0.021998456234494045, 1.6193436013223578e-06], [-0.021998456234494045, -0.9997579938434196, 0.00014720557101847266], [-1.619343601322357e-06, -0.00014720557101847266, -0.9999999891639488]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=int, default=698)
    parser.add_argument('--yaw', type=float, default=0.)
    parser.add_argument('--first-angle', type=float, default=15.)
    parser.add_argument('--last-angle', type=float, default=42.)
    parser.add_argument('--axis', type=float, nargs=3, default=[1., .23, .13])
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--search-menu', action='store_true')
    args = parser.parse_args()
    mesh = trimesh.load(G.ROOT/'objects/A4/mesh.stl', force='mesh')
    original = S.seat(mesh, REST_ROTATION)
    candidate = next(c for i, c in enumerate(G.candidates(mesh, original,
        args.candidate//39+1, max_width=.15, depths=(.10,.08,.112)))
        if i == args.candidate)
    yaw = np.eye(4)
    yaw[:3,:3] = Rotation.from_euler('z', args.yaw, degrees=True).as_matrix()
    initial = yaw@original
    candidate['opening'] = .08
    candidate['hand'] = yaw@candidate['hand']
    candidate['contacts'] = trimesh.transform_points(candidate['contacts'], yaw)
    model = G.build('A4', hand_xml=G.HERE/'assets/parallel_jaw.xml', floor_hull=True)
    assert G.geometry_check(model, initial, candidate)
    regions = WorkRegions(mesh)
    trial = S.Trial(model, mesh, initial, candidate, regions=regions)
    print('Starting integrated pickup', flush=True)
    trial.warmup()
    menu = [(args.axis, (args.first_angle, args.last_angle))]
    if args.search_menu:
        menu = [(axis, span) for span in ((6.,33.),(5.,23.),(15.,42.),(25.,52.))
                for axis in ([.23,1.,.13],[-.23,-1.,.13],[1.,.23,.13],[-1.,-.23,.13])]
    for local_axis, span in menu:
        axis = yaw[:3,:3]@np.array(local_axis)
        axis /= np.linalg.norm(axis)
        angles = np.linspace(*span, 10)
        poses = []
        for angle in angles:
            rotation = Rotation.from_rotvec(axis*np.radians(angle)).as_matrix()@trial.start[:3,:3]
            target = S.seat(mesh, rotation)
            world = trimesh.transform_points(mesh.vertices, target)
            ids = np.flatnonzero(world[:,2]<1e-8)
            com = trimesh.transform_points(mesh.center_mass[None], target)[0]
            if len(ids)!=1 or np.linalg.norm(com[:2]-world[ids[0],:2])<=.001:
                break
            if not regions.feasible(target):
                break
            hand = target@trial.relative_inverse
            # Same closed grasp at targets; no reopening/approach is performed.
            probe=mujoco.MjData(model)
            G.object_pose(model,probe,target)
            G.hand_pose(probe,hand,initialize=True)
            probe.qpos[trial.fingers]=trial.data.qpos[trial.fingers]
            mujoco.mj_forward(model,probe)
            clear=True
            for contact in probe.contact:
                ids=[int(contact.geom1),int(contact.geom2)]
                names=[model.geom(g).name for g in ids]
                if any(n.startswith('hand_geom_') for n in names) and 'floor' in names and contact.dist<-.00001:
                    clear=False
                if any(model.body(model.geom_bodyid[g]).name=='hand' for g in ids) and any(n.startswith('object_') for n in names) and contact.dist<-.00001:
                    clear=False
            if not clear:break
            poses.append(target)
        if len(poses)==10:
            try:
                for index,target in enumerate(poses,1):regions.choose(target,f'A4/pose_{index}')
            except ValueError:
                poses=[]
                continue
            break
    if len(poses)!=10:
        raise ValueError('No complete menu with a feasible work region')
    rule = dict(axis_world=axis.tolist(), angles_deg=angles.tolist(),
        preparation_yaw_deg=args.yaw, antipodal_candidate_index=args.candidate,
        candidate_depths_m=[.10,.08,.112], generator='codes/setup/a4_sequence.py')
    print(json.dumps(rule), flush=True)
    print('Starting integrated ten-target sequence', flush=True)
    checks = trial.sequence(poses)
    print(json.dumps(dict(contact_checks=checks)), flush=True)
    print('Starting full-pose KUKA checks', flush=True)
    arm = K.check(trial.history)
    trial.arm_frames, trial.arm_q, arm['scene_checks'] = K.check_scene(model, trial.history, 'A4')
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
    print(json.dumps(dict(rule=rule, arm=arm)), flush=True)
    if args.export:
        export('A4', mesh, initial, candidate, trial, poses, rule, checks, arm)
        from sequence_export import digest,write
        folder=G.ROOT/'objects/A4'
        record=json.loads((folder/'poses.json').read_text())
        record.update(generator='codes/setup/a4_sequence.py',generator_sha256=digest(__file__))
        write(folder/'poses.json',record)
        for index in range(1,11):
            p=folder/'tasks'/f'pose_{index}'
            with np.load(p/'setup.npz') as z:values={key:z[key] for key in z.files}
            values['poses_sha256']=digest(folder/'poses.json')
            np.savez_compressed(p/'setup.npz',**values)
            report=json.loads((p/'setup.json').read_text())
            report['source_snapshot_sha256']=digest(p/'setup.npz')
            write(p/'setup.json',report)
        from verify_sequences import verify
        assert verify(folder)==10


if __name__ == '__main__':
    main()
