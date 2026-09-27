"""Reproduce C1's verified rest -> ten grounded KUKA-assisted target poses.

The stable preparation and generic parallel-jaw grasp are yawed together before
integration. The free object is picked up once and never reset between targets.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import json
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
import grasp as G
import kuka_transfer as K
import sequence as S
from sequence_export import WorkRegions, export

REST_ROTATION = np.array([
    [0.9999993726831236, -0.0007918738862019272, -0.0007921925952143473],
    [-0.0007918738862019272, 0.0004027069964603336, -0.9999996053813338],
    [0.0007921925952143473, 0.9999996053813338, 0.00040207967958377777]])
CANDIDATE_INDEX = 859
YAW_DEG = 90.
ANGLES_DEG = [25., 27., 29., 30., 31., 32., 33., 34., 35., 36.]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', action='store_true')
    args = parser.parse_args()
    mesh = trimesh.load(G.ROOT/'objects/C1/mesh.stl', force='mesh')
    original = S.seat(mesh, REST_ROTATION)
    candidate = next(c for i, c in enumerate(G.candidates(mesh, original,
        CANDIDATE_INDEX//39+1, max_width=.15, depths=(.112,.118,.10)))
        if i == CANDIDATE_INDEX)
    yaw = np.eye(4)
    yaw[:3,:3] = Rotation.from_euler('z', YAW_DEG, degrees=True).as_matrix()
    initial = yaw@original
    candidate['hand'] = yaw@candidate['hand']
    candidate['contacts'] = trimesh.transform_points(candidate['contacts'], yaw)
    model = G.build('C1', hand_xml=G.HERE/'assets/parallel_jaw.xml', floor_hull=True)
    assert G.geometry_check(model, initial, candidate)
    regions = WorkRegions(mesh)
    trial = S.Trial(model, mesh, initial, candidate, regions=regions)
    print('Starting integrated pickup', flush=True)
    trial.warmup()
    axis = yaw[:3,:3]@np.array([.23,1.,.13])
    axis /= np.linalg.norm(axis)
    poses = []
    for angle in ANGLES_DEG:
        rotation = Rotation.from_rotvec(axis*np.radians(angle)).as_matrix()@trial.start[:3,:3]
        target = S.seat(mesh, rotation)
        world = trimesh.transform_points(mesh.vertices, target)
        ids = np.flatnonzero(world[:,2]<1e-8)
        com = trimesh.transform_points(mesh.center_mass[None], target)[0]
        assert len(ids)==1 and np.linalg.norm(com[:2]-world[ids[0],:2])>.001
        assert regions.feasible(target)
        test_grasp = dict(hand=target@trial.relative_inverse,
            width=2*np.mean(trial.data.qpos[trial.fingers]),
            opening=min(.08,float(np.mean(trial.data.qpos[trial.fingers]))+.012))
        assert G.geometry_check(model, target, test_grasp)
        poses.append(target)
    print('Starting integrated ten-target sequence', flush=True)
    checks = trial.sequence(poses)
    stop_frames = [min(round(c['end_time_s']/.033)-1, len(trial.history)-1) for c in checks]
    stops = Rotation.from_matrix(np.array(trial.object_history)[stop_frames,:3,:3])
    separation = min(float(np.degrees((stops[i]*stops[j].inv()).magnitude()))
        for i in range(10) for j in range(i))
    assert separation>.5, f'Actual target holds too similar: {separation} degrees'
    rule = dict(axis_world=axis.tolist(), angles_deg=ANGLES_DEG,
        preparation_yaw_deg=YAW_DEG, antipodal_candidate_index=CANDIDATE_INDEX,
        candidate_depths_m=[.112,.118,.10],
        minimum_sampled_stop_rotation_separation_deg=separation,
        stop_frame_indices=stop_frames)
    print(json.dumps(dict(contact_checks=checks, pose_selection=rule)), flush=True)
    print('Starting full-pose KUKA checks', flush=True)
    arm = K.check(trial.history)
    trial.arm_frames, trial.arm_q, arm['scene_checks'] = K.check_scene(model, trial.history, 'C1')
    print(json.dumps(dict(arm=arm)), flush=True)
    if args.export:
        export('C1', mesh, initial, candidate, trial, poses, rule, checks, arm,
            generator='codes/setup/c1_sequence.py')


if __name__ == '__main__':
    main()
